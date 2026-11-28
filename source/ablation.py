from pathlib import Path
import numpy as np
from typing import TypedDict
import csv
from config import BASE
import pandas as pd

METRIC_W=np.array([5, 3, 1], dtype=np.float32)

from collections import defaultdict

class PatientAggregator:
    def __init__(self, thresholds:tuple|None, strategy="mean", k=3, decision="weighted"):
        """
        strategy: 'mean' (baseline) ou 'topk' (melhorado)
        decision: 'weighted' (score * [5,3,1]) ou 'unweighted' (argmax puro)
        thresholds: tuplo opcional (th_present, th_unknown)
        """
        self.strategy: str= strategy
        self.k:int= k
        self.decision: str= decision
        self.thresholds: tuple|None= thresholds

    def decide(self, pr:np.ndarray, thresholds=None):
        th= thresholds if thresholds is not None else self.thresholds
        if th is not None:
            th_pres, th_unk = th
            return np.array([
                0 if p[0] >= th_pres else (1 if p[1] >= th_unk else 2)
                for p in pr
            ])
        if self.decision=="weighted":
            return (pr * METRIC_W).argmax(1) #Corrigido para retornar sempre array 1D
        else:
            return pr.argmax(1)


    def aggregate(self, probs, y_win, pids):

        #Calcula um vetor de probabilidades 1D, com combinações por classe [Present, Unknown, Absent]

        groups = defaultdict(list)
        labels = {}
        for p, y, pid in zip(probs, y_win, pids):
            #Se person_id==NaN, removemos a entrada
            if pd.isna(pid): continue
            pid_int = int(pid)

            #Safeguard: garantimos que o paciente tem uma única label
            if pid_int in labels and labels[pid_int]!=y: raise ValueError(f"Paciente com rótulos inconsistentes: {labels[pid_int]} vs {y}")

            groups[pid_int].append(p)
            labels[pid_int] = y

        ids = list(groups) #Cria uma lista de índices únicos. A LISTA NÃO ESTÁ ORDENA POR INT.

        patient_probs = []
        for pid in ids:

            wins = np.array(groups[pid])
            n_wins = len(wins)

            if self.strategy == "mean":
                # Baseline original: média aritmética simples
                p_agg = np.mean(wins, axis=0)

            elif self.strategy == "topk":

                # Versão robusta: Top-k janelas para Present e Unknown
                eff_k = min(self.k, n_wins)
                top_pres = np.mean(wins[np.argsort(wins[:, 0])[-eff_k:], 0])
                top_unk = np.mean(wins[np.argsort(wins[:, 1])[-eff_k:], 1])
                p_abs = np.mean(wins[:, 2])

                total = top_pres + top_unk + p_abs
                p_agg = np.array([top_pres, top_unk, p_abs]) / total

            else:
                raise ValueError(f"Estratégia não implementada: {self.strategy}")

            #print(pid, p_agg)
            patient_probs.append(p_agg)

        pr = np.array(patient_probs)
        y_true = np.array([labels[i] for i in ids])
        y_pred = self.decide(pr=pr)

        return y_true, y_pred, pr

def export_csv(path: Path | str, data: dict, model_name: str):
    fieldnames = ["model_name", *data.keys()]
    path = Path(path)
    has_content = path.exists() and path.stat().st_size > 0
 
    if has_content:
        with open(path, newline="", mode="r", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != fieldnames:
            raise ValueError(
                f"Colunas incompatíveis com o cabeçalho de {path.name}:\n"
                f"  ficheiro: {header}\n  linha:    {fieldnames}"
            )
 
    with open(path, newline="", mode="a", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not has_content:
            writer.writeheader()
        writer.writerow({"model_name": model_name, **data})


model_path="final_results/ablation_rnn_cv/cv_runs"
folds= [item for item in Path(model_path).iterdir() if item.is_dir()]
print(folds)

class Agg_config(TypedDict):
    strategy:str
    k:int
    mode:str

data=[
    (
        np.load(d/"probs.npy"),
        np.load(d/"y.npy"),
        np.load(d/"pids.npy", allow_pickle=True),
    )
    for d in folds
]

def wa(yt, yp):
    w = METRIC_W[yt]
    return float(np.sum(w * (yt == yp)) / np.sum(w))

agg_configs: list[Agg_config] = [
    {"strategy": str(s), "k": int(k), "mode": str(d)}
    for s, k in [("mean", 3), ("mean", 5), ("topk", 5), ("topk", 6), ("topk", 13)]
    for d in ["weighted", "unweighted"]
]


for cfg in agg_configs:
    s:str=str(cfg["strategy"])
    k:int=int(cfg["k"])
    d:str=str(cfg["mode"])

    yts, yps, per = [], [], []
    for probs, y, pid in data:
        yt, yp, _ = PatientAggregator(strategy=s, k=k, decision=d, thresholds=None).aggregate(probs, y, pid)
        yts.append(yt); yps.append(yp); per.append(wa(yt, yp))

    if not yts or not yps:
        pooled_wa = float("nan")  # or None / skip, depending on how you want to record empty runs
    else:
        pooled_wa = wa(
            yt=np.concatenate(yts),
            yp=np.concatenate(yps),
        )
    #Export results
    export_csv(
        path=BASE/"ABL_RES/rnn_ablation.csv",
        data={"strategy":s , "k": k, "decision":d, "wa":pooled_wa},
        model_name="--"
    )