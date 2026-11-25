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
import tensorflow as tf
import numpy as np

from keras import layers, models
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from datetime import datetime
from keras.losses import SparseCategoricalCrossentropy


run_date = datetime.now().strftime("%H-%M_%d-%m") 

class Config:
    # --- RNN ---
    n_mfcc: int = 13
    rnn_frame_step: int = 64 
    win_sec: float=5.0
    frame_len: int=128
    fs_target: int=1000
    seed: int=42 
    epochs: int=40
    batch_size: int=128
    K: int=5
    win = int(win_sec * fs_target)   
    # rnn_frame_step: divisão do input em sequências
    time_steps = 1 + (win - frame_len) // rnn_frame_step 
    n_features=n_mfcc*3


CLASSES = {
    0: "Present",
    1: "Unknown",
    2: "Absent"
}

METRIC_W=np.array([5, 3, 1], dtype=np.float32)



BASE= Path(".").parent.resolve()
CACHE=BASE/"data"
RESULTS_PATH=BASE/"results"
FULL_RESULTS_TABLE_PATH=RESULTS_PATH/f"{run_date}_full_output.csv"
PARAMS_TABLE_PATH=RESULTS_PATH/f"{run_date}_params.csv"

Path.mkdir(RESULTS_PATH, exist_ok=True,parents=True)
