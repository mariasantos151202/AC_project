import source.config as config
from source.config import *
from AC_project.source.patientaggregator import PatientAggregator
from patientcallback import PatientWACallback
from source.helpers import weighted_accuracy


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
            self, Xtr, ytr, Xva, yva, pva, fold:int, 
            #MODELS:dict[str, keras.Model],
            aggregator:PatientAggregator, 
            extra_metrics:list[str]|None=None,
            optimizer:keras.optimizers.Optimizer|None=None, 
            loss:keras.losses.Loss|None=None, 
            
            epochs:int=0,
            
        ):
        #Métricas adicionais a registar durante .compile. Default é [].
        extra_metrics = [] if extra_metrics is None else list(extra_metrics)

        """if any(self.model is m for m in MODELS.values()):
            print(
                f"WARNING: The current {self.model} object is being trained again"\
                "training will continue from the model's current weights."
            )"""

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
 
