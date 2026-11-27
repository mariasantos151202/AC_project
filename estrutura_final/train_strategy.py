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

    @staticmethod
    def _cm_true_pred(y_true, y_pred):
        """Matriz com linhas = verdadeiro, colunas = previsto (para as contas)."""
        m = np.zeros((3, 3), dtype=int)
        for t, p in zip(y_true, y_pred):
            m[t, p] += 1
        return m

    @staticmethod
    def compute_metrics(y_true, y_pred):
        """Métricas ao nível do doente: globais e por classe (um-contra-todos)."""
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        cm = MurmurTrainer._cm_true_pred(y_true, y_pred)
        n = cm.sum()

        tp = np.diag(cm).astype(float)
        support = cm.sum(1).astype(float)      # nº real de doentes por classe
        pred_n = cm.sum(0).astype(float)       # nº previsto por classe
        fp = pred_n - tp
        fn = support - tp
        tn = n - tp - fp - fn

        def div(a, b):
            return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)

        precision = div(tp, pred_n)
        recall = div(tp, support)              # sensibilidade
        specificity = div(tn, tn + fp)
        f1 = div(2 * precision * recall, precision + recall)

        per_class = pd.DataFrame(
            {"precisão": precision, "recall": recall, "especificidade": specificity,
             "F1": f1, "suporte": support.astype(int)},
            index=CLASSES,
        )

        w = METRIC_W[y_true]
        summary = {
            "wa": float(np.sum(w * (y_true == y_pred)) / np.sum(w)),
            "accuracy": float(tp.sum() / n),
            "balanced_accuracy": float(recall.mean()),   # média dos recalls
            "macro_f1": float(f1.mean()),
        }
        return summary, per_class

    def _print_metrics(self, y_true, y_pred):
        summary, per_class = self.compute_metrics(y_true, y_pred)
        print(f"Accuracy: {summary['accuracy']:.4f} | "
              f"Balanced acc.: {summary['balanced_accuracy']:.4f} | "
              f"Macro-F1: {summary['macro_f1']:.4f}")
        print(per_class.round(3))
        return summary, per_class

    def evaluate_and_report(self, X, y, pids, aggregator, split_name="VALIDAÇÃO", optimize_thresholds=False):
        """Avalia o modelo ao nível do paciente: WA, matriz de confusão e métricas extra."""
        probs = self.model.predict(X, batch_size=256, verbose=0)
        yt, yp, _ = aggregator.aggregate(probs, y, pids)

        w = METRIC_W[yt]
        wa = float(np.sum(w * (yt == yp)) / np.sum(w))

        print(f"\n{'='*45}")
        print(f" RESULTADOS: {split_name} | Estratégia: {aggregator.strategy.upper()}")
        print(f"{'='*45}")
        print(f"WA_murmur: {wa:.4f}")
        print(self._confusion_df(yt, yp))
        summary, per_class = self._print_metrics(yt, yp)
        self.last_report = {"summary": summary, "per_class": per_class}

        best_th = aggregator.thresholds
        if optimize_thresholds:
            best_th, best_wa = self.find_optimal_thresholds(probs, y, pids, aggregator)
            aggregator.thresholds = best_th
            _, yp_opt, _ = aggregator.aggregate(probs, y, pids)

            print(f"\n--- APÓS CALIBRAÇÃO DE LIMIARES {best_th} ---")
            print(f"WA_murmur Otimizado: {best_wa:.4f}")
            print(self._confusion_df(yt, yp_opt))
            summary_o, per_class_o = self._print_metrics(yt, yp_opt)
            self.last_report_thresholds = {"summary": summary_o, "per_class": per_class_o}

        return wa, best_th

class MurmurResNet:
    @staticmethod
    def build_extractor(frames=153, n_mels=40):
        """Constrói apenas a base ResNet50 congelada para extrair vetores de 2048 features."""
        inp = keras.Input(shape=(frames, n_mels, 1))
        x = layers.Concatenate(axis=-1)([inp, inp, inp])
        base_model = keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=x,
            pooling="avg"
        )
        base_model.trainable = False
        return keras.Model(inp, base_model.output, name="ResNet50_Extractor")

    @staticmethod
    def build_head(feature_dim=2048, n_classes=3):
        """Constrói apenas a cabeça densa que será treinada rapidamente sobre as features."""
        inp = keras.Input(shape=(feature_dim,))
        x = layers.Dropout(0.4)(inp)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(n_classes, activation="softmax")(x)
        return keras.Model(inp, out, name="ResNet50_Head")