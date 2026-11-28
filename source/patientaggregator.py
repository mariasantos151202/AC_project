import pandas as pd
import numpy as np
from collections import defaultdict
import source.config as config

 
METRIC_W=config.CFG


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

agg = PatientAggregator(
    strategy="topk",
    k=5,
    decision="unweighted",
    thresholds=None
)