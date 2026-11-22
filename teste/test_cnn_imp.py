import numpy as np
import pandas as pd
from tensorflow import keras
from cnn_structured import PatientAggregator, CLASSES, METRIC_W

# 1. Caminhos (ajusta apenas os nomes dos ficheiros/chaves se necessário)
MODEL_PATHS = [
    f"cv/fold{f}/model.keras" for f in range(5)
]
TEST_NPZ = "dados/test.npz"

# 2. Carregar dados de teste e modelos
data = np.load(TEST_NPZ, allow_pickle=True)
X_test = data["cnn2d"]
if X_test.ndim == 3:
    X_test = X_test[..., np.newaxis]
y_test = data["y"]

pids_test = pd.read_csv("dados/test_info.csv")["pid"].values

models = [keras.models.load_model(p) for p in MODEL_PATHS]


# 3. Prever janelas (média dos 5 folds) e agregar por paciente
probs_win = np.mean([m.predict(X_test, batch_size=256, verbose=0) for m in models], axis=0)

agg = PatientAggregator(strategy="topk", k=5, decision="unweighted")
yt, yp, _ = agg.aggregate(probs_win, y_test, pids_test)

# 4. Calcular WA e Matriz de Confusão 
w = METRIC_W[yt]
wa = float(np.sum(w * (yt == yp)) / np.sum(w))

cm = np.zeros((3, 3), dtype=int)
for t, p in zip(yt, yp):
    cm[p, t] += 1

print(f"WA_murmur (Teste - Ensemble 5 Folds): {wa:.4f}\n")
print(pd.DataFrame(cm, index=[f"Pred {c}" for c in CLASSES], columns=[f"True {c}" for c in CLASSES]))


#WA_murmur (Teste - Ensemble 5 Folds): 0.7852

#              True Present  True Unknown  True Absent
#Pred Present            22             3            6
#Pred Unknown             0             1            0
#Pred Absent              5             6           99