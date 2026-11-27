import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STRUCT = next((p for p in [HERE, *HERE.parents] if (p / "cnn_structured.py").exists()), None)
assert STRUCT, "não encontro o cnn_structured.py a subir a partir deste ficheiro"
sys.path.insert(0, str(STRUCT))
from train_strategy import PatientAggregator, METRIC_W

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
folds = sorted({p.parent for p in BASE.rglob("probs.npy")})
print(f"folds encontradas: {len(folds)} em {BASE}")

data = [(np.load(d/"probs.npy"), np.load(d/"y.npy"), np.load(d/"pids.npy", allow_pickle=True)) for d in folds]

def wa(yt, yp):
    w = METRIC_W[yt]
    return float(np.sum(w * (yt == yp)) / np.sum(w))

configs = [(s, k, d) for s, k in [("mean", 3), ("topk", 1), ("topk", 3), ("topk", 5)]
           for d in ["weighted", "unweighted"]]

res, lines = {}, ["agreg\tk\tdecisao\tWA_agrupado\tmedia_folds\tdp_folds"]
print(f"{'agreg':5s} {'k':>2s} {'decisão':10s} {'agrupado':>9s} {'média±dp folds':>16s}")
for s, k, d in configs:
    yts, yps, per = [], [], []
    for probs, y, pid in data:
        yt, yp, _ = PatientAggregator(strategy=s, k=k, decision=d).aggregate(probs, y, pid)
        yts.append(yt); yps.append(yp); per.append(wa(yt, yp))
    YT, YP = np.concatenate(yts), np.concatenate(yps)
    res[(s, k, d)] = (YT, YP)
    pooled = wa(YT, YP)
    kk = "-" if s == "mean" else k
    print(f"{s:5s} {kk!s:>2s} {d:10s} {pooled:9.4f} {np.mean(per):9.4f} ± {np.std(per, ddof=1):.4f}")
    lines.append(f"{s}\t{kk}\t{d}\t{pooled:.4f}\t{np.mean(per):.4f}\t{np.std(per, ddof=1):.4f}")

# bootstrap emparelhado: mesma amostra de doentes para as duas configurações
def diff_ci(a, b, n_boot=2000, seed=0):
    (yt, ya), (_, yb) = res[a], res[b]          # yt é igual nas duas (mesma ordem de doentes)
    rng = np.random.default_rng(seed)
    n = len(yt)
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        ds.append(wa(yt[i], ya[i]) - wa(yt[i], yb[i]))
    return np.mean(ds), np.percentile(ds, [2.5, 97.5])

print("\nDiferenças (A - B), IC 95% por bootstrap de doentes:")
base = ("topk", 3, "unweighted")
for other in [("mean", 3, "unweighted"), ("topk", 1, "unweighted"), ("topk", 5, "unweighted"), ("topk", 3, "weighted")]:
    m, ci = diff_ci(base, other)
    print(f"topk3/unw - {other[0]}{other[1]}/{other[2][:3]}: {m:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]")

(BASE / "cv_ablation.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nguardado em {BASE / 'cv_ablation.tsv'}")