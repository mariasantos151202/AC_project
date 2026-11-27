import numpy as np, pandas as pd
from pathlib import Path
from tensorflow import keras
from train_strategy import MurmurTrainer, PatientAggregator

RUNS = Path("cv")     

# --- k-fold (usa as previsões guardadas) ---
def report_cv(k=3):
    agg = PatientAggregator(strategy="topk", k=k, decision="unweighted")
    rows, T, P = [], [], []
    for f in range(5):
        d = RUNS / f"fold{f}"
        probs, y, pids = (np.load(d / n) for n in ("probs.npy", "y.npy", "pids.npy"))
        yt, yp, _ = agg.aggregate(probs, y, pids)
        s, _ = MurmurTrainer.compute_metrics(yt, yp)
        rows.append({"fold": f, **s}); T.append(yt); P.append(yp)
    df = pd.DataFrame(rows)
    print(df.round(4))
    print(df.drop(columns="fold").agg(["mean", "std"]).round(4))   # std com ddof=1
    s, per_class = MurmurTrainer.compute_metrics(np.concatenate(T), np.concatenate(P))
    print("\nAgregado dos 5 folds:", {a: round(b, 4) for a, b in s.items()})
    print(per_class.round(3))
    return df

df_cv = report_cv(k=5) 