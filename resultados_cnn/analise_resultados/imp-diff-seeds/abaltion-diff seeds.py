import numpy as np
from cnn_structured import PatientAggregator, METRIC_W

y = np.load("cnn/runs/y_val.npy"); p = np.load("cnn/runs/pids_val.npy")
for s in [42, 43, 44]:
    probs = np.load(f"cnn/runs/smoothed_s{s}/probs_val.npy")
    for strat in ["mean", "topk"]:
        for dec in ["weighted", "unweighted"]:
            yt, yp, _ = PatientAggregator(strategy=strat, k=3, decision=dec).aggregate(probs, y, p)
            w = METRIC_W[yt]
            print(f"seed {s} | {strat:4s} | {dec:10s} | WA = {np.sum(w*(yt==yp))/np.sum(w):.4f}")




#seed 42 | mean | weighted   | WA = 0.7296
#seed 42 | mean | unweighted | WA = 0.7852
#seed 42 | topk | weighted   | WA = 0.5407
#seed 42 | topk | unweighted | WA = 0.8296
#seed 43 | mean | weighted   | WA = 0.6963
#seed 43 | mean | unweighted | WA = 0.7889
#seed 43 | topk | weighted   | WA = 0.5296
#seed 43 | topk | unweighted | WA = 0.8444
#seed 44 | mean | weighted   | WA = 0.7926
#seed 44 | mean | unweighted | WA = 0.7963
#seed 44 | topk | weighted   | WA = 0.5593
#seed 44 | topk | unweighted | WA = 0.8222
