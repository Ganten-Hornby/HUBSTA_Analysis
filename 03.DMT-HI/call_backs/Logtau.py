

from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import torch
import wandb
import eval.eval_core_base as ecb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
import numpy as np

import matplotlib.pyplot as plt

class LogTauCallBack(Callback):
    def __init__(self, inter=10, *args, **kwargs):
        super().__init__()
        self.inter = inter
        self.plot_sample_train = 0
        self.plot_sample_test = 0

    def on_validation_epoch_end(self, trainer, pl_module):
                
        if trainer.is_global_zero and trainer.current_epoch>1:  # 检查是否为 rank 0 的进程
            
            trainer.logger.log_metrics({
                "epoch": trainer.current_epoch,
                "tau": pl_module.tau,
                })
            