import re, sys
from pathlib import Path
import numpy as np

# pasta onde estão os log_cv*.txt (procura também nas subpastas)
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
logs = sorted({p.resolve() for p in BASE.rglob("log_cv*.txt")})
print(f"BASE: {BASE} | logs encontrados: {len(logs)}")

re_head = re.compile(r"FOLD:\s*(\d+)\s*\|\s*SEED:\s*(\d+)")
re_res = re.compile(r"FOLD (\d+) \| doentes=(\d+) \| WA=([0-9.]+)")
re_epoch = re.compile(r"^Epoch (\d+)/\d+", re.M)
re_val = re.compile(r"val_wa: ([0-9.]+)")

rows = {}      # fold -> (mtime, dados): se houver logs repetidos, fica o mais recente
sem_resultado = []
for p in logs:
    txt = p.read_text(errors="ignore")
    m = re_res.search(txt)
    if not m:
        sem_resultado.append(p.name)       # corrida falhada, a correr ou antiga
        continue
    fold, n, wa = int(m.group(1)), int(m.group(2)), float(m.group(3))
    h = re_head.search(txt)
    seed = int(h.group(2)) if h else -1
    epochs = len(re_epoch.findall(txt))
    vals = [float(v) for v in re_val.findall(txt)]
    best_in = max(vals) if vals else float("nan")
    if fold not in rows or p.stat().st_mtime > rows[fold][0]:
        rows[fold] = (p.stat().st_mtime, (fold, seed, n, wa, epochs, best_in, p.name))

linhas = ["fold\tseed\tn_doentes\tWA_fold\tepochs\tmelhor_val_wa_interno\tlog"]
for f in sorted(rows):
    fold, seed, n, wa, ep, bi, name = rows[f][1]
    linhas.append(f"{fold}\t{seed}\t{n}\t{wa:.4f}\t{ep}\t{bi:.4f}\t{name}")

was = np.array([rows[f][1][3] for f in sorted(rows)])
if len(was):
    sd = np.std(was, ddof=1) if len(was) > 1 else float("nan")
    linhas.append(f"media\t\t\t{was.mean():.4f}\t\t\t")
    linhas.append(f"dp\t\t\t{sd:.4f}\t\t\t")

print("\n".join(linhas))
print(f"\nfolds com resultado: {sorted(rows)} | sem resultado: {sem_resultado or 'nenhum'}")

out = BASE / "cv_results.tsv"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(linhas) + "\n", encoding="utf-8")
print(f"guardado em {out}")


#fold    seed    n_doentes       WA_fold epochs  melhor_val_wa_interno   
#0       42      159     0.7741  22      0.7826  log_cv0.txt
#1       42      160     0.7533  26      0.7323  log_cv1.txt
#2       42      160     0.7484  16      0.6818  log_cv2.txt
#3       42      160     0.7336  17      0.8384  log_cv3.txt
#4       42      161     0.8479  21      0.8232  log_cv4.txt
#media                   0.7715
#dp                      0.0451