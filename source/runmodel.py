from source.patientaggregator import PatientAggregator
import source.config as config 
from source.config import keras
from source.murmurtrainer import MurmurTrainer
import numpy as np
from source.helpers import export_csv

def run_model(model:keras.Model, weight_mode:str, aggregator:PatientAggregator, data:dict, epochs_requested:int, fold:int, seed=None):

    seed=config.CFG.seed if seed is None else seed

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
    fold_dir = config.CV_PATH / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    #Guarda os arrays probs, y (target class), pids (ids dos pacientes) como .npy para agregação posterior ao fim de k-fold
    np.save(fold_dir / "probs.npy", test_obj["probs"]); np.save(fold_dir / "y.npy", y[test_idx]); np.save(fold_dir / "pids.npy", pid[test_idx].astype(str))

    export_csv(path=config.ACC_LOSS_PATH, data=summary, model_name=trainer.name)

    #Se correu todas as epochs ou terminou mais cedo por causa de "Early Stopping".,
    return test_obj