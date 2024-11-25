from collections.abc import Iterable
from typing import Optional, Sequence

import numpy as np
import wandb

from .callback import Callback


class WandBLogger(Callback):
    """Callback to export all metrics in epoch log of trainer to WandB.

    Args:
    + `log_dir`: Directory to save logs to. Defaults to `'./logs/wandb/'`.

    Kwargs: Additional arguments to pass into `wandb.init()`.
    """
    def __init__(self, log_dir:str='./logs/wandb/', persistent:bool=False, **kwargs):
        super().__init__()
        self.log_dir = log_dir
        self.persistent = persistent

        self.wandb_run = wandb.init(dir=self.log_dir, **kwargs)
    
        # Pre-allocate
        self.keys:Optional[Sequence[str]] = None
    
    @staticmethod
    def handle_value(k):
        is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
        if isinstance(k, str):
            return k
        elif (
            isinstance(k, Iterable)
            and not is_zero_dim_ndarray
        ):
            return f"\"[{', '.join(map(str, k))}]\""
        else:
            return k

    def on_epoch_end(self, epoch:int, logs:dict=None):
        if logs is None:
            logs = {}

        if self.keys is None:
            train_keys, val_keys = [], []
            for key in logs.keys():
                if key[0:4] == 'val_':
                    val_keys.append(key)
                else:
                    train_keys.append(key)
            self.keys = sorted(train_keys) + sorted(val_keys)

        self.wandb_run.log({k:self.handle_value(logs[k]) for k in self.keys})

    def on_train_end(self, logs:dict=None):
        if not self.persistent:
            self.wandb_run.finish()