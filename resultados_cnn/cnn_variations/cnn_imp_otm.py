from pathlib import Path
import numpy as np
import pandas as pd
from cnn_structured import MurmurCNN, PatientAggregator, MurmurTrainer, MurmurResNet
from tensorflow import keras
import os
import json
#--------------------------------------------------------------------------
#                        CARREGAR OS DADOS PRÉ-PROCESSADOS
#--------------------------------------------------------------------------

CACHE = Path("../data_analysis/cache")
d_tr = np.load(CACHE / "train.npz")
info_tr = pd.read_csv(CACHE / "train_info.csv", dtype={"pid": str})
Xtr, ytr, ptr = d_tr["cnn2d"], d_tr["y"], info_tr["pid"].values

d_va = np.load(CACHE / "val.npz")
info_va = pd.read_csv(CACHE / "val_info.csv", dtype={"pid": str})
Xva, yva, pva = d_va["cnn2d"], d_va["y"], info_va["pid"].values

#--------------------------------------------------------------------------------------------------
#                                   CNN IMPROVED COM OTIMIZAÇÃO
# Mudamos na estratégia de treino pesos suavizados, agregação dos pacientes por top k janelas
# e agora estamos também a calcular os melhores thresholds e a treinar a rede com isso
#---------------------------------------------------------------------------------------------------

model_improved = MurmurCNN.build(frames=Xtr.shape[1], n_mels=Xtr.shape[2])
trainer_imp = MurmurTrainer(model_improved, weight_mode="smoothed")
agg_imp = PatientAggregator(strategy="topk", k=3, decision="unweighted")

trainer_imp.train(Xtr, ytr, Xva, yva, pva, aggregator=agg_imp, epochs=30)

probs_imp = model_improved.predict(Xva, batch_size=128, verbose=0)
yt_i, yp_i, _ = agg_imp.aggregate(probs_imp, yva, pva)

wa_imp, th_imp = trainer_imp.evaluate_and_report(
    Xva, yva, pva,
    aggregator=agg_imp,
    split_name="VALIDAÇÃO (IMPROVED_OP)",
    optimize_thresholds= True
)
