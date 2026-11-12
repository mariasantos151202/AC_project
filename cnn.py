from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_cnn(n_mels=40, frames=153, n_classes=3): # para corresponder aos inputs do joão

    # Entrada: uma "imagem" de 40 x 153 com 1 canal.
    # É o espectrograma: eixo vertical = frequências (n_mels),
    # eixo horizontal = tempo (frames), 1 canal = preto e branco.
    inp = keras.Input(shape=(frames, n_mels, 1))
    
    #Bloco 1: procura padrões simples (16 filtros)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization(momentum=0.9)(x)   
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    #Bloco 2: procura padrões mais complexos (32 filtros)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    #Bloco 3: 64 filtros para identificar padrões mais abstratost

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inp, out)


