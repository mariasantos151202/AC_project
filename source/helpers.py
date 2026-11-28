import pandas as pd
import numpy as np
from collections import defaultdict
import source.config as config
from config import *
from AC_project.source.patientaggregator import PatientAggregator
from source.murmurtrainer import MurmurTrainer

METRIC_W=config.METRIC_W

#Matriz de confusão para a classe de interesse "Present".
#Source:https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
def conf_matrix_calc(m: pd.DataFrame|np.ndarray) -> pd.DataFrame:

    arr = m.to_numpy() if isinstance(m, pd.DataFrame) else np.asarray(m)

    TP: np.integer = arr[0,0]
    FP: np.integer = arr[1:, 0].sum()
    FN: np.integer = arr[0, 1:].sum()
    TN: np.integer = arr[1:, 1:].sum()#[1,1] +[2,1] + [1,2] + [2,2,]
    return pd.DataFrame([
        {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
        }
    ])

#Converte a matrix 3*3 para cada classe num dicionário
def cm_to_row(cm: np.ndarray) -> dict:
    """3x3 matrix ([true, predicted]) -> flat dict, one column per cell."""
    return {
        f"true_{CLASSES[i]}__pred_{CLASSES[j]}": int(cm[i, j])
        for i in range(len(CLASSES)) for j in range(len(CLASSES))
    }


#Converte matriz de confusão binária 
def added_metrics(M: pd.DataFrame) -> dict:
    row = M.iloc[0]
    TP, FP, FN = float(row["TP"]), float(row["FP"]), float(row["FN"])
    PRECISION= TP / (TP+FP)
    RECALL= TP / (TP+FN)
    F1_SCORE= 2 * (PRECISION * RECALL) / (PRECISION + RECALL)
    return dict(
        precision=PRECISION,
        f1_score=F1_SCORE,
        recall=RECALL
    )


#Definição da métrica do desafio: weighted accuracy
def weighted_accuracy(y_true, y_pred):
    # Single definition of the challenge metric, used everywhere.
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    w = METRIC_W[y_true]
    return float(np.sum(w * (y_true == y_pred)) / np.sum(w))

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

def summarize_cv(aggregator:PatientAggregator, model:keras.Model, run_id:str=run_date, seed:int=CFG.seed):

    name, n_params = model.name, model.count_params()

    def ident(fold):
        return {"run_id": run_id, "n_params": n_params, "fold": fold}

    def wa_cm_rows(row_id, y_true, y_pred=None, cm=None):
        
        cm =MurmurTrainer._confusion_matrix(y_true=y_true, y_pred=y_pred) if cm is None else cm
        patient_cm = conf_matrix_calc(cm)
        cm_rows.append({**row_id, **added_metrics(patient_cm)})
        metric_rows.append({**row_id, **added_metrics(patient_cm)})
        full_cm_rows.append({**row_id, **cm_to_row(cm)})

        return cm


    was, cms, all_true, all_pred = [], [], [], []
    wa_rows:list[dict]= []
    cm_rows:list[dict]= []
    metric_rows:list[dict]= []
    full_cm_rows: list[dict]= []
    
    for fold_file in range(CFG.K):
        
        dir = config.CV_PATH / f"fold_{fold_file}"
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
    """pooled_wa=weighted_accuracy(
        y_true=np.concatenate(all_true),
        y_pred=np.concatenate(all_pred)
    )"""
    mean_wa= float(np.mean(was))
    std_wa = float(np.std(was, ddof=1))

    export_csv(
        path=MEAN_WA_PATH, 
        data={
            "run_id": run_id, 
            "n_params": n_params, 
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


    export_rows(WA_CV_PATH, wa_rows, name)
    export_rows(PATIENT_CM_FULL_PATH, cm_rows, name)
    export_rows(METRICS_CV_PATH, metric_rows, name)
    