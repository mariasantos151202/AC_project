import source.config as config
from source.config import *
from source.helpers import weighted_accuracy

class PatientWACallback(keras.callbacks.Callback):

    def __init__(self, X_val, y_val, pids_val, aggregator):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.pids_val = pids_val
        self.aggregator = aggregator
        self.wa_history= []
        self.best_wa  = -1.0
        self.best_epoch = -1
        self.best_weights= None
        self.CFG=config.CFG

    def on_epoch_end(self, epoch, logs=None):

        if (isinstance(self.model, keras.Model)):
            probs = self.model.predict(self.X_val, batch_size=self.CFG.batch_size)

        yt, yp, _ = self.aggregator.aggregate(probs, self.y_val, self.pids_val)
        
        wa = weighted_accuracy(y_true=yt,y_pred=yp)
        
        self.wa_history.append(wa)
        
        if isinstance(logs, dict):
            logs["val_wa"] = wa
        
        if wa > self.best_wa:
            self.best_wa, self.best_epoch= wa, epoch
            self.best_weights = self.model.get_weights()
        print(f" - val_wa (paciente): {wa:.4f}")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)