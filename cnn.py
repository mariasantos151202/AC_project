from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_cnn(n_mels=64, frames=128, n_classes=3):

    # Entrada: uma "imagem" de 64 x 128 com 1 canal.
    # É o espectrograma: eixo vertical = frequências (n_mels),
    # eixo horizontal = tempo (frames), 1 canal = preto e branco.
    inp = keras.Input(shape=(n_mels, frames, 1))
    
    #Bloco 1: procura padrões simples (16 filtros)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization(momentum=0.9)(x)    
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inp, out)


#esta tudo explicado no notas

# -------------Função para fazer os dados artificais-----------------------------

rng = np.random.default_rng(42)

def make_data(n=600, n_mels=64, frames=128):
    y = rng.integers(0, 3, size=n)                  # 0=Absent, 1=Unknown, 2=Present
    X = rng.normal(0, 1, size=(n, n_mels, frames, 1)).astype("float32")
    for i, c in enumerate(y):
        if c == 1:      # banda de energia em frequências baixas
            X[i, 10:20, :, 0] += 1.5
        elif c == 2:    # banda em frequências altas, só na 2ª metade
            X[i, 40:50, 64:, 0] += 1.5
    return X, y

X_train, y_train = make_data(600)
X_val, y_val = make_data(150)


model = build_cnn()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()
# dar o fit normal
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=10, batch_size=32)

probs = model.predict(X_val[:5])
print(np.round(probs, 2))       # probabilidades [Absent, Unknown, Present]
print("reais:", y_val[:5])
print(model.evaluate(X_train, y_train, verbose=0))
print(model.evaluate(X_val, y_val, verbose=0))