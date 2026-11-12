from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(42)

CLASSES = ["Present", "Unknown", "Absent"]  # 0, 1, 2
METRIC_W = np.array([5.0, 3.0, 1.0])


# =====================================================================
# 1. ARQUITETURA DA REDE
# =====================================================================
class MurmurCNN:
    @staticmethod
    def build(frames=153, n_mels=40, n_classes=3):
        inp = keras.Input(shape=(frames, n_mels, 1))

        # Bloco 1: 16 filtros
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)

        # Bloco 2: 32 filtros
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)

        # Bloco 3: 64 filtros
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        out = layers.Dense(n_classes, activation="softmax")(x)

        return keras.Model(inp, out, name="Custom_CNN2D")


# =====================================================================
# 2. AGREGADOR POR PACIENTE (Baseline vs. Melhorado)
# =====================================================================
class PatientAggregator:
    def __init__(self, strategy="mean", k=3, decision="weighted", thresholds=None):
        """
        strategy: 'mean' (baseline) ou 'topk' (melhorado)
        decision: 'weighted' (score * [5,3,1]) ou 'unweighted' (argmax puro)
        thresholds: tuplo opcional (th_present, th_unknown)
        """
        self.strategy = strategy
        self.k = k
        self.decision = decision
        self.thresholds = thresholds

    def aggregate(self, probs, y_win, pids):
        groups = defaultdict(list)
        labels = {}
        for p, y, pid in zip(probs, y_win, pids):
            groups[pid].append(p)
            labels[pid] = y

        ids = list(groups)
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
                raise ValueError(f"Estratégia desconhecida: {self.strategy}")

            patient_probs.append(p_agg)

        pr = np.array(patient_probs)
        y_true = np.array([labels[i] for i in ids])

        # Decisão final
        if self.thresholds is not None:
            th_pres, th_unk = self.thresholds
            preds = [0 if p[0] >= th_pres else (1 if p[1] >= th_unk else 2) for p in pr]
            y_pred = np.array(preds)
        else:
            score = pr * METRIC_W if self.decision == "weighted" else pr
            y_pred = score.argmax(axis=1)

        return y_true, y_pred, pr


# =====================================================================
# 3. GESTOR DE TREINO E CALLBACKS
# =====================================================================
class PatientWACallback(keras.callbacks.Callback):
    def __init__(self, X_val, y_val, pids_val, aggregator):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.pids_val = pids_val
        self.aggregator = aggregator

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.X_val, batch_size=128, verbose=0)
        yt, yp, _ = self.aggregator.aggregate(probs, self.y_val, self.pids_val)
        w = METRIC_W[yt]
        wa = float(np.sum(w * (yt == yp)) / np.sum(w))
        logs["val_wa"] = wa
        print(f" - val_wa (paciente): {wa:.4f}")


class MurmurTrainer:
    def __init__(self, model, weight_mode="smoothed"):
        """weight_mode: 'standard' (inverso clássico), 'smoothed' (raiz quadrada) ou None"""
        self.model = model
        self.weight_mode = weight_mode

    def compute_weights(self, y_train):
        if self.weight_mode is None:
            return None
        counts = np.bincount(y_train, minlength=3).astype(float)
        if self.weight_mode == "standard":
            w = counts.sum() / (3.0 * counts)
        elif self.weight_mode == "smoothed":
            w = np.sqrt(counts.sum() / (3.0 * counts))
            w = w / w.mean()
        return {i: float(w[i]) for i in range(3)}

    def train(self, Xtr, ytr, Xva, yva, pva, aggregator, epochs=40, batch_size=64):
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = [
            PatientWACallback(Xva, yva, pva, aggregator),
            keras.callbacks.EarlyStopping(monitor="val_wa", mode="max", patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
        ]

        return self.model.fit(
            Xtr, ytr,
            validation_data=(Xva, yva),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=self.compute_weights(ytr),
            callbacks=callbacks,
            verbose=2
        )
    def find_optimal_thresholds(self, probs_val, y_val, pids_val, aggregator):
        """Encontra o par (th_present, th_unknown) que maximiza o WA na validação."""
        yt, _, pr = aggregator.aggregate(probs_val, y_val, pids_val)
        
        best_wa = -1.0
        best_th = (0.33, 0.33)
        
        for th_pres in np.linspace(0.15, 0.75, 61):
            for th_unk in np.linspace(0.15, 0.65, 51):
                preds = np.array([
                    0 if p[0] >= th_pres else (1 if p[1] >= th_unk else 2)
                    for p in pr
                ])
                w = METRIC_W[yt]
                wa = float(np.sum(w * (yt == preds)) / np.sum(w))
                if wa > best_wa:
                    best_wa = wa
                    best_th = (round(float(th_pres), 3), round(float(th_unk), 3))
                    
        return best_th, best_wa

    def evaluate_and_report(self, X, y, pids, aggregator, split_name="VALIDAÇÃO", optimize_thresholds=False):
        """Avalia o modelo ao nível do paciente, imprime WA e matriz de confusão."""
        probs = self.model.predict(X, batch_size=256, verbose=0)
        yt, yp, _ = aggregator.aggregate(probs, y, pids)
        
        w = METRIC_W[yt]
        wa = float(np.sum(w * (yt == yp)) / np.sum(w))
        
        print(f"\n{'='*45}")
        print(f" RESULTADOS: {split_name} | Estratégia: {aggregator.strategy.upper()}")
        print(f"{'='*45}")
        print(f"WA_murmur: {wa:.4f}")
        print(self._confusion_df(yt, yp))
        
        best_th = aggregator.thresholds
        if optimize_thresholds:
            best_th, best_wa = self.find_optimal_thresholds(probs, y, pids, aggregator)
            # Atualiza o agregador com os melhores limiares encontrados
            aggregator.thresholds = best_th
            _, yp_opt, _ = aggregator.aggregate(probs, y, pids)
            
            print(f"\n--- APÓS CALIBRAÇÃO DE LIMIARES {best_th} ---")
            print(f"WA_murmur Otimizado: {best_wa:.4f}")
            print(self._confusion_df(yt, yp_opt))
            
        return wa, best_th

    @staticmethod
    def _confusion_df(y_true, y_pred):
        m = np.zeros((3, 3), dtype=int)
        for t, p in zip(y_true, y_pred):
            m[p, t] += 1
        return pd.DataFrame(
            m,
            index=[f"Pred {c}" for c in CLASSES],
            columns=[f"True {c}" for c in CLASSES]
        )