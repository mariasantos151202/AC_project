import sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras

HERE = Path(__file__).resolve().parent                 # structured/cnn
sys.path.insert(0, str(HERE.parent))                   # para encontrar structured/cnn_structured.py
from cnn_structured import MurmurCNN, PatientAggregator, MurmurTrainer

CACHE = HERE.parent.parent / "data_analysis" / "cache"

# -----------------------------------------
#         CARREGAR OS DADOS
#----------------------------------------------
d_tr = np.load(CACHE / "train.npz")
info_tr = pd.read_csv(CACHE / "train_info.csv", dtype={"pid": str})
Xtr, ytr, ptr = d_tr["cnn2d"], d_tr["y"], info_tr["pid"].values

d_va = np.load(CACHE / "val.npz")
info_va = pd.read_csv(CACHE / "val_info.csv", dtype={"pid": str})
Xva, yva, pva = d_va["cnn2d"], d_va["y"], info_va["pid"].values

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
keras.utils.set_random_seed(seed)
OUT = HERE / "runs" / f"smoothed_s{seed}"
OUT.mkdir(parents=True, exist_ok=True)

model_improved = MurmurCNN.build(frames=Xtr.shape[1], n_mels=Xtr.shape[2])
trainer_imp = MurmurTrainer(model_improved, weight_mode="smoothed")
agg_imp = PatientAggregator(strategy="topk", k=3, decision="unweighted")

history = trainer_imp.train(Xtr, ytr, Xva, yva, pva, aggregator=agg_imp, epochs=30)

# guardar LOGO, antes de qualquer outra coisa que possa falhar
model_improved.save(OUT / "model.keras")
probs_imp = model_improved.predict(Xva, batch_size=128, verbose=0)
np.save(OUT / "probs_val.npy", probs_imp)
np.save(OUT.parent / "y_val.npy", yva)
np.save(OUT.parent / "pids_val.npy", np.array(pva, dtype=str))
pd.DataFrame(history.history).to_csv(OUT / "history.csv", index=False)

wa_imp, _ = trainer_imp.evaluate_and_report(
    Xva, yva, pva, aggregator=agg_imp,
    split_name=f"VALIDAÇÃO (IMPROVED, seed {seed})",
    optimize_thresholds=False)