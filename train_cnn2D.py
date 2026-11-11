"""
Liga o pré-processamento (cache/*.npz) à CNN (cnn.py) e avalia ao nível do paciente.
 
Pré-requisitos:
  1. Ter corrido a secção 4 do notebook de pré-processamento (gera cache/).
  2. No cnn.py, a parte dos dados artificiais (make_data ... evaluate) tem de estar
     dentro de `if __name__ == "__main__":`, senão importar o ficheiro corre esse treino.
 
Correr (a partir da pasta onde estão cnn.py e este ficheiro):
  python train_cnn.py
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
 
from cnn import build_cnn
 
CACHE = Path("data_analysis/cache")     # <- ajustar se a cache estiver noutro sítio
CLASSES = ["Present", "Unknown", "Absent"]          # índices 0, 1, 2 (igual ao pré-processamento)
METRIC_W = np.array([5.0, 3.0, 1.0])
RUN_TEST = False                         # só pôr True no FIM, depois de fixar tudo
EPOCHS, BATCH = 40, 64

keras.utils.set_random_seed(42)

# ---------------------------------------------------------------- dados
def load_split(name):
    d = np.load(CACHE / f"{name}.npz")
    info = pd.read_csv(CACHE / f"{name}_info.csv", dtype={"pid": str})
    return d["cnn2d"], d["y"], info["pid"].values
 
 
Xtr, ytr, ptr = load_split("train")
Xva, yva, pva = load_split("val")
print("train:", Xtr.shape, "val:", Xva.shape)        # (N, 153, 40, 1)
 
 
# ---------------------------------------------------------------- métrica / agregação
def weighted_accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    w = METRIC_W[y_true]
    return float(np.sum(w * (y_true == y_pred)) / np.sum(w))
 
 
def confusion(y_true, y_pred):
    m = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[p, t] += 1                                   # linhas = previsto, colunas = verdadeiro
    return pd.DataFrame(m, index=[f"Pred {c}" for c in CLASSES],
                        columns=[f"True {c}" for c in CLASSES])
 
 
def to_patient(probs, y_win, pids, decision="weighted"):
    """Média das probabilidades das janelas de cada paciente -> 1 previsão por paciente."""
    groups, labels = defaultdict(list), {}
    for p, y, pid in zip(probs, y_win, pids):
        groups[pid].append(p)
        labels[pid] = y
    ids = list(groups)
    pr = np.stack([np.mean(groups[i], axis=0) for i in ids])
    score = pr * METRIC_W if decision == "weighted" else pr
    return np.array([labels[i] for i in ids]), score.argmax(1)
 
 
class PatientWA(keras.callbacks.Callback):
    """WA ao nível do paciente na validação, no fim de cada época."""
    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(Xva, batch_size=128, verbose=0)
        yt, yp = to_patient(probs, yva, pva)
        logs["val_wa"] = weighted_accuracy(yt, yp)
        print(f" - val_wa (paciente): {logs['val_wa']:.4f}")
 
 
# ---------------------------------------------------------------- modelo e treino
frames, n_mels = Xtr.shape[1], Xtr.shape[2]
model = build_cnn(n_mels=n_mels, frames=frames)      # input = (frames, n_mels, 1)
 
counts = np.bincount(ytr, minlength=3).astype(float)
class_weight = {i: float(counts.sum() / (3 * counts[i])) for i in range(3)}
print("class weights:", class_weight)
 
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
 
model.fit(
    Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH,
    class_weight=class_weight,
    callbacks=[
        PatientWA(),                                   # tem de vir antes do EarlyStopping
        keras.callbacks.EarlyStopping(monitor="val_wa", mode="max", patience=8,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ],
    verbose=2,
)
model.save("cnn_best.keras")
 
 
# ---------------------------------------------------------------- avaliação
def report(name, X, y, pids):
    probs = model.predict(X, batch_size=128, verbose=0)
    yt, yp = to_patient(probs, y, pids)
    print(f"\n=== {name} (nível de paciente) ===")
    print("WA_murmur:", round(weighted_accuracy(yt, yp), 4))
    print(confusion(yt, yp))
 
 
report("VALIDAÇÃO", Xva, yva, pva)
if RUN_TEST:
    Xte, yte, pte = load_split("test")
    report("TESTE", Xte, yte, pte)