from pathlib import Path
from cnn_structured import PatientAggregator
import numpy as np
import pandas as pd
from tensorflow import keras

CACHE = Path("cache")
d = np.load(CACHE / "hidden.npz")
info = pd.read_csv(CACHE / "hidden_info.csv", dtype={"pid": str})
X, pids = d["cnn2d"], info["pid"].values

agg = PatientAggregator(strategy="topk", k=3, decision="unweighted")
y_dummy = np.zeros(len(X), dtype=int)

fold_probs = []
for f in range(5):
    m = keras.models.load_model(f"cv/fold{f}/model.keras")
    probs = m.predict(X, batch_size=128, verbose=0)
    _, _, pr = agg.aggregate(probs, y_dummy, pids)
    fold_probs.append(pr)

pr_ens = np.mean(fold_probs, axis=0)
ids = list(dict.fromkeys(pids))

pred = (pr_ens * [1, 1, 1]).argmax(axis=1)  # ou os teus limiares calibrados
names = np.array(["Present", "Unknown", "Absent"])
preds_map = dict(zip(ids, names[pred]))

# Preencher o ficheiro oficial de resultados mantendo a ordem e colunas exatas
results_path = Path("../CirCor_test_students/PLX_GYY_test_results.csv")
results_df = pd.read_csv(results_path)
col_id, col_murmur = results_df.columns[0], results_df.columns[1]

results_df[col_murmur] = results_df[col_id].astype(str).map(preds_map)
assert results_df[col_murmur].notna().all(), "Atenção: há pacientes sem previsão!"
results_df.to_csv("PLX_GYY_test_results.csv", index=False)
print("Previsões guardadas com sucesso!")