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
# 1. ARQUITETURA DA CNN2D
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
