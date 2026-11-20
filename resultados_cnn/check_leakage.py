import numpy as np, pandas as pd
from pathlib import Path

C = Path("../data_analysis/cache")
stats = {}
for n in ["train", "val", "test"]:
    X = np.load(C / f"{n}.npz")["cnn2d"]
    info = pd.read_csv(C / f"{n}_info.csv", dtype={"pid": str})
    assert len(X) == len(info), f"{n}: npz e csv com tamanhos diferentes"
    stats[n] = (X.size, float(X.mean()), float(X.std()))
    wm = X.reshape(len(X), -1).mean(axis=1)   # média de cada janela
    print(n, X.shape, "média global %.4f | std %.4f | média por janela: min %.3f max %.3f"
          % (stats[n][1], stats[n][2], wm.min(), wm.max()))
    del X

# média ponderada dos 3 splits
N = sum(s[0] for s in stats.values())
print("média dos 3 juntos: %.6f" % (sum(s[0]*s[1] for s in stats.values()) / N))