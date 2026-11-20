import numpy as np
from cnn_structured import PatientAggregator, METRIC_W

seeds = [42, 43, 44]
y = np.load("cnn/runs/y_val.npy")
p = np.load("cnn/runs/pids_val.npy")

def wa(probs):
    agg = PatientAggregator(strategy="topk", k=3, decision="unweighted")
    yt, yp, _ = agg.aggregate(probs, y, p)
    w = METRIC_W[yt]
    return np.sum(w * (yt == yp)) / np.sum(w)

probs = {s: np.load(f"cnn/runs/smoothed_s{s}/probs_val.npy") for s in seeds}
was = [wa(probs[s]) for s in seeds]

for s, v in zip(seeds, was):
    print(f"seed {s}: WA = {v:.4f}")
print(f"média = {np.mean(was):.4f} | desvio-padrão = {np.std(was, ddof=1):.4f}")
print(f"ensemble (média das probs) = {wa(np.mean(list(probs.values()), axis=0)):.4f}")

#seed 42: WA = 0.8296
#seed 43: WA = 0.8444
#seed 44: WA = 0.8222
#média = 0.8321 | desvio-padrão = 0.0113
#ensemble (média das probs) = 0.8185