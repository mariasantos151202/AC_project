import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # WSL: só grava imagens, sem janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------- CONFIG -----------------------------
BASE = Path("cv")                           # pasta com fold0/, fold1/, ... (cada uma com o log)
OUT = Path("figs"); OUT.mkdir(exist_ok=True)
TEST_WA = 0.7852                           # WA_murmur do ensemble no teste
MIN_FOLDS = 3                              # só desenha a média onde há >= MIN_FOLDS folds
KEYS = ["accuracy", "loss", "val_accuracy", "val_loss", "val_wa", "learning_rate"]

# ----------------------------- LEITURA -----------------------------
def parse_text(text):
    hist = {k: [] for k in KEYS}
    wa_lines = re.findall(r"val_wa \(paciente\): ([\d.]+)", text)
    for line in text.splitlines():
        if re.match(r"\s*\d+/\d+ - ", line):
            for k in KEYS:
                m = re.search(rf"(?<!\w){k}: ([\d.e+-]+)", line)
                if m:
                    hist[k].append(float(m.group(1)))
    if not hist["val_wa"] and wa_lines:
        hist["val_wa"] = [float(x) for x in wa_lines]
    return {k: np.array(v, float) for k, v in hist.items()}

def parse_csv(path):
    df = pd.read_csv(path)
    return {k: df[k].values.astype(float) for k in KEYS if k in df.columns}

def load_fold(folder):
    files = sorted(list(folder.rglob("*.txt")))
    for f in files:
        h = parse_csv(f) if f.suffix == ".csv" else parse_text(f.read_text(errors="ignore"))
        if len(h.get("loss", [])) > 0:
            print(f"{folder.name}: {f.name} -> {len(h['loss'])} épocas")
            return h
    raise FileNotFoundError(f"Sem log válido em {folder}")

fold_dirs = sorted(d for d in BASE.iterdir() if d.is_dir() and not d.name.startswith((".", "figs", "__")))
histories = [load_fold(d) for d in fold_dirs]
K = len(histories)
n_max = max(len(h["loss"]) for h in histories)
ep_all = np.arange(1, n_max + 1)

# ----------------------------- AUXILIARES -----------------------------
def pad(a, n):
    a = np.asarray(a, float)[:n]                      # trunca se for mais comprido
    return np.pad(a, (0, n - len(a)), constant_values=np.nan)   # preenche se for mais curto

def stack(key):
    return np.array([pad(h[key], n_max) for h in histories])

def mean_sd(arr):
    valid = np.sum(~np.isnan(arr), axis=0) >= MIN_FOLDS
    mu = np.where(valid, np.nanmean(arr, 0), np.nan)
    sd = np.where(valid, np.nanstd(arr, 0), np.nan)
    return mu, sd

def draw_mean_curves(ax, metric, title):
    keys = [(metric, "treino", "C0"), (f"val_{metric}", "validação", "C1")]
    for key, label, c in keys:
        arr = stack(key)
        for r in arr:
            ax.plot(ep_all, r, color=c, alpha=.15, lw=1)
        mu, sd = mean_sd(arr)
        ax.plot(ep_all, mu, color=c, lw=2, label=label)
        ax.fill_between(ep_all, mu - sd, mu + sd, color=c, alpha=.15)
    ax.set(title=title, xlabel="Época"); ax.grid(alpha=.3); ax.legend()

def draw_wa_mean(ax):
    arr = stack("val_wa")
    for r in arr:
        ax.plot(ep_all, r, color="C2", alpha=.15, lw=1)
    mu, sd = mean_sd(arr)
    ax.plot(ep_all, mu, color="C2", lw=2, label="média dos folds")
    ax.fill_between(ep_all, mu - sd, mu + sd, color="C2", alpha=.15)
    ax.axhline(TEST_WA, ls=":", color="r", label=f"teste (ensemble) = {TEST_WA:.3f}")
    ax.set(title="WA_murmur (paciente, validação)", xlabel="Época", ylim=(0.4, 1))
    ax.grid(alpha=.3); ax.legend()

def draw_wa_per_fold(ax):
    for i, h in enumerate(histories):
        ep = np.arange(1, len(h["val_wa"]) + 1)
        b = int(np.argmax(h["val_wa"]))
        ax.plot(ep, h["val_wa"], color=f"C{i}", alpha=.8, label=f"fold {i}")
        ax.scatter(ep[b], h["val_wa"][b], color=f"C{i}", s=60, zorder=3, edgecolor="k")
    ax.set(title="WA por fold (● = melhor época)", xlabel="Época", ylim=(0.4, 1))
    ax.grid(alpha=.3); ax.legend(ncol=K, fontsize=8)

def draw_wa_bars(ax):
    best = [np.max(h["val_wa"]) for h in histories]
    ax.bar(range(K), best, color="C2")
    ax.axhline(np.mean(best), ls="--", color="k", label=f"média = {np.mean(best):.3f} ± {np.std(best):.3f}")
    ax.axhline(TEST_WA, ls=":", color="r", label=f"teste = {TEST_WA:.3f}")
    ax.set(title="Melhor val WA por fold", xlabel="Fold", ylim=(0.4, 1), xticks=range(K))
    ax.legend(fontsize=8)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("gravado:", OUT / name)

# ----------------------------- FIGURAS -----------------------------
# 1) Curvas médias (loss, accuracy, WA)
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
draw_mean_curves(ax[0], "loss", "Loss")
draw_mean_curves(ax[1], "accuracy", "Accuracy (por janela)")
draw_wa_mean(ax[2])
save(fig, "kfold_curves.png")

# 2) Mesmas curvas cortadas à época mínima comum (sem viés de early stopping)
n_common = min(len(h["loss"]) for h in histories)
n_max_backup = n_max
n_max = n_common
ep_all = np.arange(1, n_common + 1)
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
draw_mean_curves(ax[0], "loss", f"Loss (até época {n_common})")
draw_mean_curves(ax[1], "accuracy", f"Accuracy (até época {n_common})")
draw_wa_mean(ax[2])
save(fig, "kfold_curves_common.png")
n_max = n_max_backup
ep_all = np.arange(1, n_max + 1)

# 3) WA por fold com melhor época + barras
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
draw_wa_per_fold(ax[0])
draw_wa_bars(ax[1])
save(fig, "kfold_wa.png")

# 4) Figura única para os slides
fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(2, 3)
draw_mean_curves(fig.add_subplot(gs[0, 0]), "loss", "Loss")
draw_mean_curves(fig.add_subplot(gs[0, 1]), "accuracy", "Accuracy (por janela)")
draw_wa_mean(fig.add_subplot(gs[0, 2]))
draw_wa_per_fold(fig.add_subplot(gs[1, :2]))
draw_wa_bars(fig.add_subplot(gs[1, 2]))
fig.suptitle(f"CNN 2D — validação cruzada ({K} folds)", fontsize=14)
save(fig, "kfold_slide.png")

# 5) Uma figura por fold (loss / accuracy / WA com melhor época)
for i, h in enumerate(histories):
    ep = np.arange(1, len(h["loss"]) + 1)
    b = int(np.argmax(h["val_wa"]))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(ep, h["loss"], label="treino"); ax[0].plot(ep, h["val_loss"], label="validação")
    ax[0].set(title="Loss", xlabel="Época")
    ax[1].plot(ep, h["accuracy"], label="treino"); ax[1].plot(ep, h["val_accuracy"], label="validação")
    ax[1].set(title="Accuracy (por janela)", xlabel="Época")
    ax[2].plot(np.arange(1, len(h["val_wa"]) + 1), h["val_wa"], color="C2", label="val WA")
    ax[2].axvline(b + 1, ls="--", color="gray", label=f"melhor época ({b+1}: {h['val_wa'][b]:.3f})")
    ax[2].set(title="WA_murmur (paciente)", xlabel="Época", ylim=(0.4, 1))
    for a in ax:
        a.grid(alpha=.3); a.legend()
    fig.suptitle(f"Fold {i}")
    save(fig, f"curves_fold{i}.png")

    # 6) Um ficheiro por gráfico
    panels = {
        "loss":        lambda ax: draw_mean_curves(ax, "loss", "Loss"),
        "accuracy":    lambda ax: draw_mean_curves(ax, "accuracy", "Accuracy (por janela)"),
        "wa_media":    draw_wa_mean,
        "wa_por_fold": draw_wa_per_fold,
        "wa_barras":   draw_wa_bars,
    }
    for name, fn in panels.items():
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        fn(ax)
        save(fig, f"kfold_{name}.png")

# ----------------------------- RESUMO -----------------------------
best = [np.max(h["val_wa"]) for h in histories]
print("\nMelhor val WA por fold:", [round(float(b), 4) for b in best])
print(f"Média ± desvio: {np.mean(best):.4f} ± {np.std(best):.4f}  |  teste (ensemble): {TEST_WA:.4f}")