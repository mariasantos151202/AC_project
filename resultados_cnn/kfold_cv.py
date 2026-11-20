import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow import keras

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "data_analysis" / "cache").exists())
STRUCT = next(p for p in [HERE, *HERE.parents] if (p / "cnn_structured.py").exists())
sys.path.insert(0, str(STRUCT))
from cnn_structured import MurmurCNN, MurmurTrainer, PatientAggregator, METRIC_W

K = 5
fold = int(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

CACHE = ROOT / "data_analysis" / "cache"
RUNS = STRUCT / "cnn" / "runs"
print("CACHE:", CACHE, "| existe:", CACHE.exists(), "| FOLD:", fold, "| SEED:", seed)

def load(name):
    d = np.load(CACHE / f"{name}.npz")
    info = pd.read_csv(CACHE / f"{name}_info.csv", dtype={"pid": str})
    return d["cnn2d"], d["y"], info["pid"].values

Xa, ya, pa = load("train"); Xb, yb, pb = load("val")
X = np.concatenate([Xa, Xb]); y = np.concatenate([ya, yb]); pid = np.concatenate([pa, pb])
del Xa, Xb

# fold de avaliação (mesma partição em todas as corridas: random_state fixo)
outer = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=0)
tr_idx, te_idx = list(outer.split(np.zeros(len(y)), y, pid))[fold]
assert not set(pid[tr_idx]) & set(pid[te_idx]), "doentes repetidos entre treino e fold"

# validação interna (early stopping), só com doentes do treino
inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=1)
a, b = next(inner.split(np.zeros(len(tr_idx)), y[tr_idx], pid[tr_idx]))
fit_idx, in_idx = tr_idx[a], tr_idx[b]

keras.utils.set_random_seed(seed)
model = MurmurCNN.build(frames=X.shape[1], n_mels=X.shape[2])
trainer = MurmurTrainer(model, weight_mode="smoothed")
agg = PatientAggregator(strategy="topk", k=3, decision="unweighted")
trainer.train(X[fit_idx], y[fit_idx], X[in_idx], y[in_idx], pid[in_idx],
              aggregator=agg, epochs=30)

# guardar logo
OUT = RUNS / "cv" / f"fold{fold}"
OUT.mkdir(parents=True, exist_ok=True)
model.save(OUT / "model.keras")
probs = model.predict(X[te_idx], batch_size=128, verbose=0)
np.save(OUT / "probs.npy", probs)
np.save(OUT / "y.npy", y[te_idx])
np.save(OUT / "pids.npy", pid[te_idx].astype(str))

yt, yp, _ = agg.aggregate(probs, y[te_idx], pid[te_idx])
w = METRIC_W[yt]
wa = float(np.sum(w * (yt == yp)) / np.sum(w))
print(f"FOLD {fold} | doentes={len(yt)} | WA={wa:.4f}")
open(RUNS / "results.tsv", "a").write(f"cv_fold{fold}\t{seed}\ttopk\tunweighted\t{wa:.4f}\n")