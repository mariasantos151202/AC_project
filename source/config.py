# Configuration imports 
from collections import defaultdict
from datetime import datetime, timezone
import os
from pathlib import Path
import time
import sys
from pathlib import Path
import numpy as np, pandas as pd
from tensorflow import keras


from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import csv

from keras import layers, models, losses
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from datetime import datetime
from keras.losses import SparseCategoricalCrossentropy
from itertools import islice

class Config:
    # --- RNN ---
    n_mfcc: int = 13
    rnn_frame_step: int = 64 
    win_sec: float=5.0
    frame_len: int=128
    fs_target: int=1000
    seed: int=42 
    epochs: int=30
    batch_size: int=64
    K: int=5
    win = int(win_sec * fs_target)   
    # rnn_frame_step: divisão do input em sequências
    time_steps = 1 + (win - frame_len) // rnn_frame_step 
    n_features=n_mfcc*3

#Global variables
run_date = datetime.now().strftime("%H-%M-%S_%d-%m") 

METRIC_W = np.array([5.0, 3.0, 1.0])
CFG=Config()
CLASSES = {
    0: "Present",
    1: "Unknown",
    2: "Absent"
}

#Project paths
BASE:Path=Path(__file__).resolve().parents[1]
CACHE=Path("C:/Users/user/Desktop/Code_Projects/AC_II_project/final_branch/AC_project/data")


RESULTS_PATH=BASE/"run_results"
RESULTS_RUN_PATH=RESULTS_PATH/f"{run_date}"
CV_PATH=RESULTS_RUN_PATH/"cv_log"

#Param table: Aggregator configuration for the run
PARAMS_PATH=RESULTS_RUN_PATH/"params.csv"

#
ACC_LOSS_PATH=CV_PATH/"fold_acc_loss.csv"
METRICS_CV_PATH=CV_PATH/"fold_f1-score.csv"
PATIENT_CM_FULL_PATH=CV_PATH/"fold_full_cm.csv"
WA_CV_PATH=CV_PATH/"fold_wa.csv"

#
MEAN_WA_PATH=RESULTS_RUN_PATH/"mean_wa.csv"
POOL_CM_PATH=RESULTS_RUN_PATH/"pool_cm.csv"

Path.mkdir(RESULTS_RUN_PATH, exist_ok=True, parents=True)
Path.mkdir(CV_PATH, exist_ok=True, parents=True)