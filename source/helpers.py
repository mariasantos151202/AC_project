import pandas as pd
import numpy as np
from collections import defaultdict
import source.config as config
from source.config import *
#from source.patientaggregator import PatientAggregator

METRIC_W=config.METRIC_W

#Em conformidade com a matriz de confusão do sklearn.
   
def confusion_matrix_values(y_true, y_pred):
    m = np.zeros((3, 3), dtype=int)

    for TRUE, PREDICTED in zip(y_true, y_pred):
        m[TRUE, PREDICTED] += 1
    
    return m

def _to_df(m: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        m,
        index=[f"True {c}" for c in CLASSES.values()],
        columns=[f"Pred {c}" for c in CLASSES.values()]
    )

def _confusion_df(y_true, y_pred):
    return _to_df(confusion_matrix_values(y_true, y_pred))


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

#Converte a matrix 3*3 para cada classe num dicionário
def cm_to_row(cm: np.ndarray) -> dict:
    """3x3 matrix ([true, predicted]) -> flat dict, one column per cell."""
    return {
        f"true_{CLASSES[i]}__pred_{CLASSES[j]}": int(cm[i, j])
        for i in range(len(CLASSES)) for j in range(len(CLASSES))
    }


#Converte matriz de confusão binária 
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


#Definição da métrica do desafio: weighted accuracy
def weighted_accuracy(y_true, y_pred):
    # Single definition of the challenge metric, used everywhere.
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    w = METRIC_W[y_true]
    return float(np.sum(w * (y_true == y_pred)) / np.sum(w))

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

def export_rows(path:Path|str, data:list[dict], model_name:str):
    
    if not data:
        return
    keys=list(data[0])
    
    for r in data:
        if list(r.keys())!=keys:
            raise ValueError(f"A linha a ser escrita não tem a mesma estrutura do header:{Path(path).name}")

    with open(path, newline="", mode="w") as f:
        #data.keys é um dicionário unidimensional. O unpacking é com *, não **. #dict_items([('params', 199615), ('epochs', 1)])
        writer = csv.DictWriter(f, fieldnames=["model_name", *keys] )
        writer.writeheader()
        for r in data:
            writer.writerow({"model_name":model_name, **r})

#Exportar para .md (jupyter). Formatação das tabelas em markdown. para este ciclo.
def md_table_build(csv_file:Path|str) -> str:
    """POOL_CM_PATH --> model_name, **cm_to_row --> build em .md;
    #METRICAS ADICIONAIS"""

    df = pd.read_csv(csv_file, sep=",")
    header=df.columns.tolist()
    
    #Build header and separator lines
    lines=[
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |"
    ]

    #Build row content 
    for row in df.values:
        lines.append("| " + " | ".join(v for v in row) + " |")

    return "\n".join(lines)
