#This implementation is based on Andrew McDonald's team repository.

import pandas as pd
import numpy as np

#Matriz de confusão para a classe de interesse "Present".
#Source:https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
def conf_matrix_calc(m: pd.DataFrame|np.ndarray) -> pd.DataFrame:

    arr = m.to_numpy() if isinstance(m, pd.DataFrame) else np.asarray(m)

    TP: np.integer = arr[0,0]
    FP: np.integer = arr[1:, 0].sum()
    FN: np.integer = arr[0, 1:].sum()
    TN: np.integer = arr[1:, 1:].sum()#[1,1] +[2,1] + [1,2] + [2,2,]
    return pd.DataFrame([
        {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
        }
    ])

#Recebe 
def added_metrics(M: pd.DataFrame) -> dict:
    row = M.iloc[0]
    TP, FP, FN = float(row["TP"]), float(row["FP"]), float(row["FN"])
    PRECISION= TP / (TP+FP)
    RECALL= TP / (TP+FN)
    F1_SCORE= 2 * (PRECISION * RECALL) / (PRECISION + RECALL)
    return dict(
        precision=PRECISION,
        f1_score=F1_SCORE,
        recall=RECALL
    )
