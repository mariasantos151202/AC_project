
from pathlib import Path
import numpy as np
import pandas as pd
from cnn_structured import MurmurCNN, PatientAggregator, MurmurTrainer, MurmurResNet
from tensorflow import keras
import os
import json
#--------------------------------------------------------------------------
#                        CARREGAR OS DADOS PRÉ-PROCESSADOS
#--------------------------------------------------------------------------

CACHE = Path("../data_analysis/cache")
d_tr = np.load(CACHE / "train.npz")
info_tr = pd.read_csv(CACHE / "train_info.csv", dtype={"pid": str})
Xtr, ytr, ptr = d_tr["cnn2d"], d_tr["y"], info_tr["pid"].values

d_va = np.load(CACHE / "val.npz")
info_va = pd.read_csv(CACHE / "val_info.csv", dtype={"pid": str})
Xva, yva, pva = d_va["cnn2d"], d_va["y"], info_va["pid"].values


model_baseline = MurmurCNN.build(frames=Xtr.shape[1], n_mels=Xtr.shape[2])
trainer_base = MurmurTrainer(model_baseline, weight_mode="standard")
agg_base = PatientAggregator(strategy="mean", decision="weighted")

trainer_base.train(Xtr, ytr, Xva, yva, pva, aggregator=agg_base, epochs=30)

probs_base = model_baseline.predict(Xva, batch_size=128, verbose=0)
yt_b, yp_b, _ = agg_base.aggregate(probs_base, yva, pva)

# Imprime imediatamente o WA_murmur e a matriz de confusão da Baseline
wa_base, _ = trainer_base.evaluate_and_report(
    Xva, yva, pva,
    aggregator=agg_base,
    split_name="VALIDAÇÃO (BASELINE)",
    optimize_thresholds=False
)

#--------------------------------------------------------------------------------------------------
#                                   CNN BASELINE
# 3 blocos (16, 32, 64) com Max Pooling + duas denses + camada de output com GlobalAveragePooling
# cerca de 28 000 paramatros treinaveis, o que correpsonde mais ou menos a um ciclo cardiaco
#---------------------------------------------------------------------------------------------------



def run_experiment(name, weights="smoothed", seed=42, epochs=30):
    keras.utils.set_random_seed(seed)
    out = f"runs/{name}"
    os.makedirs(out, exist_ok=True)

    model_baseline = MurmurCNN.build(frames=Xtr.shape[1], n_mels=Xtr.shape[2])
    trainer_base = MurmurTrainer(model_baseline, weight_mode="standard")
    agg_base = PatientAggregator(strategy="mean", decision="weighted")

    history = trainer_base.train(Xtr, ytr, Xva, yva, pva, aggregator=agg_base, epochs=epochs)

    model_baseline.save(f"{out}/model.keras")
    np.save(f"{out}/probs_val.npy", model_baseline.predict(Xva, batch_size=128, verbose=0))
    np.save(f"{out}/y_val.npy", yva)
    np.save(f"{out}/pids_val.npy", np.array(pva))
    pd.DataFrame(history.history).to_csv(f"{out}/history.csv", index=False)
    json.dump({"weights": weights, "seed": seed, "epochs": epochs}, open(f"{out}/args.json", "w"))