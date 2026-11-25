from sources.config import *

CFG=Config()

class MyModel(keras.Model):
    def __init__(self, name:str, **kwargs):
        super().__init__(name=name, **kwargs)

McDonald_model:MyModel=models.Sequential(
    [
        layers.Input(shape=(CFG.time_steps,CFG.n_features)),
        layers.Bidirectional(layers.GRU(units=64, return_sequences=True)), #return_sequences=True; as camadas anteriores são passadas para layers seguintes. Essas camadas são passadas como sequências.
        layers.Bidirectional(layers.GRU(units=64, return_sequences=True)),
        layers.Bidirectional(layers.GRU(units=64)), #O output final é ATIVAÇÃO (computado a partir de pesos e inputs).
        layers.Dropout(0.1),
        #Bloco 2: : two-layer NN, fully-connected, c/ funções de ativação Tanh, c/tamanho 60 e 40
        layers.Dense(60, activation='tanh'),
        layers.Dropout(0.1),
        layers.Dense(40, activation='tanh'),
        #OUTPUT
        layers.Dense(len([key for key in CLASSES.values()]), activation='softmax') # 5 segmentation labels per time step
    ],  
    name="McDonald"
)
