#!/usr/bin/env python
# coding: utf-8

# # Imports e funções globais
# 

# In[ ]:


from sources.config import *
from sources import models_designs
from sources import metrics
from itertools import islice

CFG = Config()

RESULTS:   dict[str, dict] = {}   # run name -> summary metrics (incl. test WA)
PARAMS:    dict[str, dict] = {}   # run name -> configuration used for the run
MODELS:    dict[str, keras.Model] = {}              # run name -> trained model

#Definição da métrica do desafio: weighted accuracy
def weighted_accuracy(y_true, y_pred):
    # Single definition of the challenge metric, used everywhere.
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    w = METRIC_W[y_true]
    return float(np.sum(w * (y_true == y_pred)) / np.sum(w))


# # PatientAggregator

# In[ ]:


from collections import defaultdict

class PatientAggregator:
    def __init__(self, thresholds:tuple|None, strategy="mean", k=3, decision="weighted"):
        """
        strategy: 'mean' (baseline) ou 'topk' (melhorado)
        decision: 'weighted' (score * [5,3,1]) ou 'unweighted' (argmax puro)
        thresholds: tuplo opcional (th_present, th_unknown)
        """
        self.strategy: str= strategy
        self.k:int= k
        self.decision: str= decision
        self.thresholds: tuple|None= thresholds

    def decide(self, pr:np.ndarray, thresholds=None):
        th= thresholds if thresholds is not None else self.thresholds
        if th is not None:
            th_pres, th_unk = th
            return np.array([
                0 if p[0] >= th_pres else (1 if p[1] >= th_unk else 2)
                for p in pr
            ])
        if self.decision=="weighted":
            return (pr * METRIC_W).argmax(1) #Corrigido para retornar sempre array 1D
        else:
            return pr.argmax(1)


    def aggregate(self, probs, y_win, pids):

        #Calcula um vetor de probabilidades 1D, com combinações por classe [Present, Unknown, Absent]

        groups = defaultdict(list)
        labels = {}
        for p, y, pid in zip(probs, y_win, pids):
            #Se person_id==NaN, removemos a entrada
            if pd.isna(pid): continue
            pid_int = int(pid)

            #Safeguard: garantimos que o paciente tem uma única label
            if pid_int in labels and labels[pid_int]!=y: raise ValueError(f"Paciente com rótulos inconsistentes: {labels[pid_int]} vs {y}")

            groups[pid_int].append(p)
            labels[pid_int] = y

        ids = list(groups) #Cria uma lista de índices únicos. A LISTA NÃO ESTÁ ORDENA POR INT.

        patient_probs = []
        for pid in ids:

            wins = np.array(groups[pid])
            n_wins = len(wins)

            if self.strategy == "mean":
                # Baseline original: média aritmética simples
                p_agg = np.mean(wins, axis=0)

            elif self.strategy == "topk":

                # Versão robusta: Top-k janelas para Present e Unknown
                eff_k = min(self.k, n_wins)
                top_pres = np.mean(wins[np.argsort(wins[:, 0])[-eff_k:], 0])
                top_unk = np.mean(wins[np.argsort(wins[:, 1])[-eff_k:], 1])
                p_abs = np.mean(wins[:, 2])

                total = top_pres + top_unk + p_abs
                p_agg = np.array([top_pres, top_unk, p_abs]) / total

            else:
                raise ValueError(f"Estratégia não implementada: {self.strategy}")

            #print(pid, p_agg)
            patient_probs.append(p_agg)

        pr = np.array(patient_probs)
        y_true = np.array([labels[i] for i in ids])
        y_pred = self.decide(pr=pr)

        return y_true, y_pred, pr

#== Patient Aggregator ==#
agg = PatientAggregator(strategy="topk",k=3, decision="unweighted", thresholds=None)




# # Treinar modelo criado na deteção de sopros cardíacos

# In[ ]:


class PatientWACallback(keras.callbacks.Callback):

    def __init__(self, X_val, y_val, pids_val, aggregator):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.pids_val = pids_val
        self.aggregator = aggregator
        self.wa_history= []
        self.best_wa  = -1.0
        self.best_epoch = -1
        self.best_weights= None
        self.CFG=Config()

    def on_epoch_end(self, epoch, logs=None):

        if (isinstance(self.model, keras.Model)):
            probs = self.model.predict(self.X_val, batch_size=self.CFG.batch_size)

        yt, yp, _ = self.aggregator.aggregate(probs, self.y_val, self.pids_val)
        
        wa = weighted_accuracy(y_true=yt,y_pred=yp)
        
        self.wa_history.append(wa)
        
        if isinstance(logs, dict):
            logs["val_wa"] = wa
        
        if wa > self.best_wa:
            self.best_wa, self.best_epoch= wa, epoch
            self.best_weights = self.model.get_weights()
        print(f" - val_wa (paciente): {wa:.4f}")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


class MurmurTrainer:
    #Global class variable assignment

    def __init__(self, model:keras.Model, weight_mode="smoothed"):
        """weight_mode: 'standard' (inverso clássico), 'smoothed' (raiz quadrada) ou None"""
        self.model = model
        self.name = model.name
        self.weight_mode = weight_mode
        self.CFG=Config()

    def compute_weights(self, y_train):
        if self.weight_mode is None:
            return None
        #Contagem do número de ocorrências de cada valor no array
        counts = np.bincount(np.asarray(y_train, dtype=int), minlength=len(CLASSES)).astype(float)
        if self.weight_mode == "standard":
            w = counts.sum() / (len(CLASSES) * counts)
        elif self.weight_mode == "smoothed":
            w = np.sqrt(counts.sum() / (3.0 * counts))
            w = w / w.mean()
        return {i: float(w[i]) for i in range(3)}


    def train(
            self, Xtr, ytr, Xva, yva, pva, fold:int, aggregator:PatientAggregator, 
            extra_metrics:list[str]|None=None,
            optimizer:keras.optimizers.Optimizer|None=None, 
            loss:keras.losses.Loss|None=None, 
            epochs:int=0
        ):
        #Métricas adicionais a registar durante .compile. Default é [].
        extra_metrics = [] if extra_metrics is None else list(extra_metrics)

        if any(self.model is m for m in MODELS.values()):
            print(
                f"WARNING: The current {self.model} object is being trained again"\
                "training will continue from the model's current weights."
            )

        optimizer = keras.optimizers.Adam(learning_rate=1e-3) if optimizer is None else optimizer
        loss = keras.losses.SparseCategoricalCrossentropy() if loss is None else loss

        #Learning rate follow-up
        """csv_logger=keras.callbacks.CSVLogger(
            filename=str(TRAINING_LOG_PATH),
            separator=",",
            append=False
        )"""

        if (isinstance(self.model, keras.Model)):
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=["accuracy", *extra_metrics]
            )

        wa_callback = PatientWACallback(Xva, yva, pva, aggregator)
        #Tracking learning rate per epoch:
        callbacks = [
            wa_callback,
            keras.callbacks.EarlyStopping(monitor="val_wa", mode="max", patience=8),# wa_callback.on_train_end recupera sempre os melhores pesos.
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
        ]

        t0 = time.time()

        history=self.model.fit(
            Xtr, ytr,
            validation_data=(Xva, yva),
            epochs=epochs,
            batch_size=self.CFG.batch_size,
            class_weight=self.compute_weights(ytr),
            callbacks=callbacks,
            verbose=1
        )
        t1 = time.time()-t0

        #== SAVE TRAINING METRICS ==#
        h = history.history
        h["val_wa"]=list(wa_callback.wa_history)
        be = wa_callback.best_epoch #Retorna os valores treino/validação da melhor época registada até agora.

        summary = dict(
            fold=fold,
            params=self.model.count_params(),
            epochs=len(h["loss"]),
            train_acc=h["accuracy"][-1],
            val_acc=h["val_accuracy"][-1],
            best_val_acc=max(h["val_accuracy"]),
            val_loss=h["val_loss"][-1],
            min_val_loss=min(h["val_loss"]),
            gap=h["accuracy"][-1] - h["val_accuracy"][-1],
            fit_runtime=round(t1, 1),
            best_epoch=be+1, #Check if useful
            best_val_wa=wa_callback.best_wa, #Check if useful
        )

        meta = dict(
            fold=fold,
            epochs_requested=epochs,
            batch_size=self.CFG.batch_size,
            weight_mode=self.weight_mode,
            optimizer=type(optimizer).__name__,
            loss=type(loss).__name__,
            aggregator_strategy=aggregator.strategy,
            aggregator_decision=aggregator.decision,
        )

        return summary, meta

    @staticmethod
    def find_optimal_thresholds(probs_val, y_val, pids_val, aggregator):
        """Encontra o par (th_present, th_unknown) que maximiza o WA na validação."""
        yt, _, pr = aggregator.aggregate(probs_val, y_val, pids_val)

        best_wa, best_th = -1.0, (0.33, 0.33)

        for th_pres in np.linspace(0.15, 0.75, 61):
            for th_unk in np.linspace(0.15, 0.65, 51):
                wa = weighted_accuracy(yt, aggregator.decide(pr, thresholds=(th_pres, th_unk)))
                if wa > best_wa:
                    best_wa = wa
                    best_th = (round(float(th_pres), 3), round(float(th_unk), 3))

        return best_th, best_wa

    """
    Create confusion matrix of model 

    """

    # Avaliação do paciente
    def evaluate_and_report(self, X, y, pids, aggregator, thresholds:tuple[float, float]|None=None, split_name="VALIDAÇÃO", optimize_thresholds=False):
            """Avalia o modelo ao nível do paciente, imprime WA e matriz de confusão."""

            probs = self.model.predict(X, batch_size=self.CFG.batch_size)
            yt, yp, pr = aggregator.aggregate(probs, y, pids)
            wa = weighted_accuracy(yt, yp)
            cm = self._confusion_matrix(y_true=yt,y_pred=yp)
            
            th, wa_th, cm_th = thresholds, None, None
            if optimize_thresholds:
                th, _ = self.find_optimal_thresholds(probs, y, pids, aggregator)
                # Atualiza o agregador com os melhores limiares encontrados
            if th is not None:
                #Determina se predicted results sem thresholds é melhor do que com thresholds.
                yp_th = aggregator.decide(pr, thresholds=th)             
                wa_th=weighted_accuracy(yt, yp_th)
                cm_th = self._confusion_matrix(yt, yp_th)
                print(self._to_df(cm_th))

            return {
                "wa": wa, "wa_thresholds": wa_th, 
                "optimal_wa": wa_th if wa_th is not None else wa,
                "thresholds": th,"cm": cm, 
                "cm_thresholds": cm_th, 
                "probs": probs
            }

    """
    CLASSES = {
        0: "Present",
        1: "Unknown",
        2: "Absent"
    }    
    """
    #Em conformidade com a matriz de confusão do sklearn.
    @staticmethod
    def _confusion_matrix(y_true, y_pred):
        m = np.zeros((3, 3), dtype=int)

        for TRUE, PREDICTED in zip(y_true, y_pred):
            m[TRUE, PREDICTED] += 1
        
        return m

    @staticmethod
    def _to_df(m: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            m,
            index=[f"True {c}" for c in CLASSES.values()],
            columns=[f"Pred {c}" for c in CLASSES.values()]
        )

    @staticmethod
    def _confusion_df(y_true, y_pred):
        return MurmurTrainer._to_df(MurmurTrainer._confusion_matrix(y_true, y_pred))
 


# ## Data loading and Hold-out validation

# In[ ]:


def load_split(architecture, name, CACHE):
    cache = np.load(CACHE / f"{name}.npz")
    info = pd.read_csv(CACHE / f"{name}_info.csv", dtype={"pid": str})
    #Extrai do zip .npz o numpy array 
    return cache[f"{architecture}"].astype("float32"), cache["y"].astype(int), info["pid"].to_numpy(dtype=object)
    #Não verificamos pid porque todas as entradas têm o id do paciente.


Xtr, ytr, ptr = load_split(architecture="cnn", name="train", CACHE=Path(CACHE))
Xva, yva, pva = load_split(architecture="cnn", name="val", CACHE=Path(CACHE))

# Concatenar dados de treino e validação
X:np.ndarray=np.concatenate([Xtr, Xva])
y:np.ndarray=np.concatenate([ytr, yva])
pid:np.ndarray=np.concatenate([ptr, pva])

"""
Hold-out: escolhemos 1 fold externa e utilizamos as 6 folds internas.

"""


def split_fold(y, pid, k, fold=0):

    #Outer fold: Data splitter. Object will split fed data into K sets, grouped by patient id
    outer_split = StratifiedGroupKFold(
        n_splits=k,
        shuffle=True,
        random_state=0
    )

    inner_split=StratifiedGroupKFold(
        n_splits=6, 
        shuffle=True, 
        random_state=1
    )

    dummy_X = np.zeros(len(y)) #Quantas samples (casos registados) existem.

    #Gera K folds ((train_idx, test_idx)), que são tuplos de np.Array, com dados de features(X), target class(y) e código do paciente(pid)
    #Split apenas das posições (indexes) do dummy array, 
    #isslice avança por todas as folds, ao invés de retornar apenas a fold=0
    train_idx, test_idx = next(islice(outer_split.split(dummy_X, y, pid), fold, None))


    #Garantir que não há data leakage de identidades: um paciente está presente na validação e no treino.
    assert not set(pid[train_idx]) & set(pid[test_idx]), "Data leakage: doentes repetidos entre treino e fold"

    #Inner splitter object

    #Split indexes of only training array
    dummy_X=np.zeros(len(train_idx))

    #Posições dentro do array de treino (train_idx)
    #inner_split.split --> objeto Iterator gera folds step by step. 
    # next() --> Guarda a primeira iteração do Iterator
    #Evita construir "n_splits" e guardar em memória
    #Gera posições relativas dentro de train_idx para o set de treino

    fit_pos, val_pos = next(inner_split.split(dummy_X, y[train_idx], pid[train_idx]))

    #Recolher índices dos registos (linhas do array) para 2 efeitos:
        #FIT: rows used for weigth training
        #IN: rows used for val_wa/epoch
    fit_idx, in_idx = train_idx[fit_pos], train_idx[val_pos]

    return fit_idx, in_idx, test_idx


# In[ ]:


"""
Calcula a probabilidade final do paciente para aquela sequência.
Thresholds: (th_present), (th_unknown)
    (th_present) p[0]: Probabilidade final mínima para classificar como "Sopro cardíaco presente" (0). 
    (th_unknown) p[1]: Probabilidae final m~´inima para classificar como "Sopro cardíaco não foi encontrado" (1)
    - Se o tuplo for fornecido
    - Se o tuplo não fornecido
        - Adiciona viés às classes se atributo self.decision da classe Aggregator for "weighted"
        - A probabilidade será o maior valor registado.    

y_pred = np.array(preds) é a coleção de valores previstos. Gerada por decide().
"""



# In[ ]:


#Exportar resultados em tabelas .csv em modo "w", garantindo que um crash não gera um ficheiro de output pela metade ou algum registo é sobrescrito.
import csv

def export_csv(path: Path | str, data: dict, model_name: str):
    fieldnames = ["model_name", *data.keys()]
    path = Path(path)
    has_content = path.exists() and path.stat().st_size > 0
 
    if has_content:
        with open(path, newline="", mode="r", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != fieldnames:
            raise ValueError(
                f"Colunas incompatíveis com o cabeçalho de {path.name}:\n"
                f"  ficheiro: {header}\n  linha:    {fieldnames}"
            )
 
    with open(path, newline="", mode="a", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not has_content:
            writer.writeheader()
        writer.writerow({"model_name": model_name, **data})


def export_rows(path:Path|str, data:list[dict], model_name:str):
    
    if not data:
        return
    keys=list(data[0])
    
    for r in data:
        if list(r.keys())!=keys:
            raise ValueError(f"A linha a ser escrita não tem a mesma estrutura do header:{Path(path).name}")

    with open(path, newline="", mode="w") as f:
        #data.keys é um dicionário unidimensional. O unpacking é com *, não **. #dict_items([('params', 199615), ('epochs', 1)])
        writer = csv.DictWriter(f, fieldnames=["model_name", *keys] )
        writer.writeheader()
        for r in data:
            writer.writerow({"model_name":model_name, **r})


# In[ ]:


def run_model(model:keras.Model, weight_mode:str, aggregator:PatientAggregator, data:dict, epochs_requested:int, fold:int, seed=None):

    seed=CFG.seed if seed is None else seed

    #LOAD DATA NDARRAY for model
    X, y, pid = data["X"], data["y"], data["pid"]
    fit_idx, in_idx, test_idx = data["fit_idx"], data["in_idx"], data["test_idx"]

    trainer=MurmurTrainer(
        model=model, #keras.Model
        weight_mode=weight_mode,
    )

    keras.utils.set_random_seed(seed)

    summary, meta = trainer.train(
        Xtr=X[fit_idx].astype("float32"),
        ytr=y[fit_idx],
        Xva=X[in_idx].astype("float32"),
        yva=y[in_idx],
        pva=pid[in_idx],
        aggregator=aggregator,
        epochs=epochs_requested,
        fold=fold
    )

    #Thresholds calculadas durante as 6 folds internas de treino+validação
    #validation_obj: wa, wa_thresholds, thresholds
    val_obj = trainer.evaluate_and_report(
        X=X[in_idx],
        y=y[in_idx],
        pids=pid[in_idx],
        aggregator=aggregator,
        split_name="VALIDAÇÃO",
        optimize_thresholds=True
    )

    #Thresholds calculadas no treino usadas no teste
    test_obj = trainer.evaluate_and_report(
        X=X[test_idx],
        y=y[test_idx],
        pids=pid[test_idx],
        aggregator=aggregator,
        split_name="TESTE",
        optimize_thresholds=False,
        thresholds= val_obj["thresholds"]
    )

    #Guardar previsões da fold
    fold_dir = CV_RUNS / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    #Guarda os arrays probs, y (target class), pids (ids dos pacientes) como .npy para agregação posterior ao fim de k-fold
    np.save(fold_dir / "probs.npy", test_obj["probs"]); np.save(fold_dir / "y.npy", y[test_idx]); np.save(fold_dir / "pids.npy", pid[test_idx].astype(str))


    export_csv(path=FULL_RESULTS_TABLE_PATH, data=summary, model_name=trainer.name)
    export_csv(path=PARAMS_TABLE_PATH, data=meta, model_name=trainer.name)

    #Se correu todas as epochs ou terminou mais cedo por causa de "Early Stopping".,
    return test_obj


# # Run models

# In[ ]:


#Converte a matrix 3*3 para cada classe num dicionário
def cm_to_row(cm: np.ndarray) -> dict:
    """3x3 matrix ([true, predicted]) -> flat dict, one column per cell."""
    return {
        f"true_{CLASSES[i]}__pred_{CLASSES[j]}": int(cm[i, j])
        for i in range(len(CLASSES)) for j in range(len(CLASSES))
    }


## Agregar todos os outputs/paciente: patient-level confusion matrix, weighted accuracy, 
#Usamos o mesmo aggregator: agg(topk, k=3, unweighted)
agg = PatientAggregator(strategy="topk", k=3, decision="unweighted", thresholds=None)

def summarize_cv(aggregator:PatientAggregator, model:keras.Model, run_id:str=run_date, seed:int=CFG.seed):

    name, n_params = model.name, model.count_params()

    def ident(fold):
        return {"run_id": run_id, "n_params": n_params, "fold": fold}

    def wa_cm_rows(row_id, y_true, y_pred=None, cm=None):
        
        cm =MurmurTrainer._confusion_matrix(y_true=y_true, y_pred=y_pred) if cm is None else cm
        patient_cm = metrics.conf_matrix_calc(cm)
        cm_rows.append({**row_id, **metrics.added_metrics(patient_cm)})
        metric_rows.append({**row_id, **metrics.added_metrics(patient_cm)})
        full_cm_rows.append({**row_id, **cm_to_row(cm)})

        return cm


    was, cms, all_true, all_pred = [], [], [], []
    wa_rows:list[dict]= []
    cm_rows:list[dict]= []
    metric_rows:list[dict]= []
    full_cm_rows: list[dict]= []
    
    for fold_file in range(CFG.K):
        
        dir = CV_RUNS / f"fold_{fold_file}"
        probs, y, pids= (np.load(dir/n) for n in ("probs.npy", "y.npy", "pids.npy"))
        y_true, y_predicted, _ = aggregator.aggregate(probs, y, pids)

        #Calcular weighted accuracy/fold
        wa=weighted_accuracy(y_true=y_true, y_pred=y_predicted)
        wa_rows.append({**ident(fold=fold_file), "weighted_accuracy":wa})

        #Calcular patient-level confusion matrix/fold
        cm = wa_cm_rows(row_id=ident(fold=fold_file), y_true=y_true, y_pred=y_predicted)

        was.append(wa)
        cms.append(cm)
        all_true.append(y_true)
        all_pred.append(y_predicted)


    #calcula Weighted Accuracy, com respetiva média e desvio padrão.
    pooled_wa=weighted_accuracy(
        y_true=np.concatenate(all_true),
        y_pred=np.concatenate(all_pred)
    )
    mean_wa= float(np.mean(was))
    std_wa = float(np.std(was, ddof=1))

    export_csv(
        path=POOL_WA_PATH, 
        data={
            "run_id": run_id, 
            "n_params": n_params, 
            "pooled_wa":pooled_wa,
            "mean_wa":mean_wa, 
            "std_wa":std_wa, 
        },
        model_name= name
    )

    pooled_cm=np.sum(cms, axis=0)
    export_csv(
        path=POOL_CM_PATH,
        data={
            "run_id":run_id,
            "n_params": n_params,
            **cm_to_row(pooled_cm)
        },
        model_name=name
    )


    export_rows(WA_PATH, wa_rows, name)
    export_rows(PATIENT_CM_PATH, cm_rows, name)
    export_rows(ADDED_METRICS_PATH, metric_rows, name)
    export_rows(PATIENT_CM_FULL_PATH, full_cm_rows, name)

#=========#

"""
Calling designed ANN.

"""

keras.utils.set_random_seed(CFG.seed)
from sources import models_designs

for fold in range(CFG.K):

    fit_idx, in_idx, test_idx= split_fold(y=y, pid=pid, k=CFG.K, fold=fold)

    DATA = dict(
        X=X,
        y=y, 
        pid=pid,
        fit_idx=fit_idx,
        in_idx=in_idx,
        test_idx=test_idx
    )

    keras.utils.set_random_seed(CFG.seed)

    #Para cada nova fold, corremos o modelo novamente (completamente limpo)

    model=keras.models.clone_model( models_designs.Custom_CNN2D)

    run_model(
        model=model,    
        weight_mode="smoothed",
        aggregator=agg,
        data=DATA,
        epochs_requested=CFG.epochs,
        seed=CFG.seed,
        fold=fold
    )

summarize_cv(aggregator=agg, model=models_designs.Custom_CNN2D)


# %%
