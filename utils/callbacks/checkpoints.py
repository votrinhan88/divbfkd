from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np
import torch

from .callback import Callback


class ModelCheckpoint(Callback):
    """Callback to save a model or its weights, based on Keras's implementation 
    and refactored to a simpler version.

    Args:
    + `target`: Target model to save. Defaults to `None`.
    + `filepath`: Path to save the model. Defaults to `None`, skip to save to   \
        './model_checkpoint.pt'.
    + `monitor`: The metric name to monitor. Defaults to `"val_loss"`.
    + `verbose`: Verbosity mode: `0` | `1`. `0`: silent, `1`: display messages  \
        when the callback takes an action. Defaults to `0`.
    + `save_best_only`: Flag to only save when the model is considered the best,\
        and the latest best model according to the quantity monitored will not  \
        be overwritten. Defaults to `False`.
    + `save_state_dict_only`: Flag to only save the model's state dict, else the\
        whole model is saved. Defaults to `True`.
    + `mode`: Saving mode based on monitor: `'auto'` | `'min'` | `'max'`.       \
        `'auto'`: minimum value for loss or maximum value for accuracy, `'min'`:\
        minimum value, `'max'`: maximum value. Defaults to `"auto"`.
    + `save_freq`: Number of epochs to check and save model. Defaults to `1`.
    + `initial_value_threshold`: Initial "best" value of the metric to be       \
        monitored. Defaults to `None`.
    """
    def __init__(self,
        target:Optional[Union[torch.nn.Module, torch.optim.Optimizer]]=None,
        filepath:Optional[str]=None,
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        save_state_dict_only:bool = True,
        mode: str = "auto",
        save_freq:int=1,
        initial_value_threshold=None,
    ):
        super().__init__()
        self.target = target
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_state_dict_only = save_state_dict_only
        self.mode = mode
        self.save_freq = save_freq
        
        if self.filepath is None:
            self.filepath = './model_checkpoint.pt'

        self.epochs_since_last_save = 0
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self.best = initial_value_threshold

        if self.mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"ModelCheckpoint mode {self.mode} is unknown, fallback to "
                "'auto' mode.",
            )
            self.mode = "auto"

        if self.mode == "min":
            self.monitor_op = np.less
            if self.best is None:
                self.best = np.Inf
        elif self.mode == "max":
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.Inf
        else:
            if "acc" in self.monitor:
                self.monitor_op = np.greater
                if self.best is None:
                    self.best = -np.Inf
            elif "loss" in self.monitor:
                self.monitor_op = np.less
                if self.best is None:
                    self.best = np.Inf
            else:
                raise ValueError(
                    f"Unrecognized `mode` for monitor '{self.monitor}'. Please "
                    "specify manually."
                )

    def on_train_begin(self, logs=None):
        if self.target is None:
            try:
                self.target = self.host.model
            except:
                raise ValueError(
                    "Unrecognized `target` model, please specify manually."
                )

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1

        if epoch % self.save_freq == 0:
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def _save_model(self, epoch, batch, logs:Optional[dict]=None):
        """Saves the model.
        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
              is set to `epoch`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if (
            isinstance(self.save_freq, int)
            or self.epochs_since_last_save >= self.period
        ):
            self.epochs_since_last_save = 0

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        if self.verbose > 0:
                            warnings.warn(
                                f"Can save best model only with `{self.monitor}` "
                                "available, skipping"
                            )
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print(
                                    f"\nEpoch {epoch}: {self.monitor} improved "
                                    f"from {self.best:.4f} to {current:.4f}, "
                                    f"saving model to {self.filepath}"
                                )
                            self.best = current
                            if self.save_state_dict_only:
                                torch.save(obj=self.target.state_dict(), f=self.filepath)
                            else:
                                torch.save(obj=self.target, f=self.filepath)
                        else:
                            if self.verbose > 0:
                                print(
                                    f"\nEpoch {epoch}: `{self.monitor}` did not"
                                    f" improve from {self.best:.4f}"
                                )
                else:
                    if self.verbose > 0:
                        print(
                            f"\nEpoch {epoch}: saving model to {self.filepath}"
                        )
                    if self.save_state_dict_only:
                        torch.save(obj=self.target.state_dict(), f=self.filepath)
                    else:
                        torch.save(obj=self.target, f=self.filepath)

            except IsADirectoryError:
                raise IOError(
                    "Please specify a non-directory filepath for "
                    "ModelCheckpoint. Filepath used is an existing "
                    f"directory: {self.filepath}"
                )


class Checkpoint(Callback):
    """Callback to save an attribute of a model.

    Args:
    + `attribute`.
    + `filepath`. Defaults to `None`.
    + `monitor`. Defaults to `"val_loss"`.
    + `verbose`. Defaults to `0`.
    + `save_best_only`. Defaults to `False`.
    + `mode`. Defaults to `"auto"`.
    + `save_freq`. Defaults to `1`.
    + `save_last`. Defaults to `True`.
    + `initial_value_threshold`. Defaults to `None`.
    + `suffix`. Defaults to `False`.
    """
    def __init__(self,
        attribute:str,
        filepath:Optional[str]=None,
        monitor:str="val_loss",
        verbose:int=0,
        save_best_only:bool=False,
        mode:str="auto",
        save_freq:int=1,
        save_last:bool=True,
        initial_value_threshold=None,
        suffix:bool=False,
    ):
        super().__init__()
        self.attribute = attribute
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.save_last = save_last
        self.suffix = suffix
        
        if self.filepath is None:
            self.filepath = './checkpoint.pt'

        self.epochs_since_last_save = 0
        self.best = initial_value_threshold

        if self.mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"Checkpoint mode {self.mode} is unknown, fallback to "
                "'auto' mode.",
            )
            self.mode = "auto"

        if self.mode == "min":
            self.monitor_op = np.less
            if self.best is None:
                self.best = np.Inf
        elif self.mode == "max":
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.Inf
        else:
            if "acc" in self.monitor:
                self.monitor_op = np.greater
                if self.best is None:
                    self.best = -np.Inf
            elif "loss" in self.monitor:
                self.monitor_op = np.less
                if self.best is None:
                    self.best = np.Inf
            else:
                raise ValueError(
                    f"Unrecognized `mode` for monitor '{self.monitor}'. Please "
                    "specify manually."
                )

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1

        if epoch % self.save_freq == 0:
            self.save(epoch=epoch, logs=logs)
    
    def on_train_end(self, logs=None):
        if self.save_last:
            self.save(epoch='final', logs=logs)

    def save(self, epoch, logs:Optional[dict]=None):
        """Saves the target attribute.
        Args:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        assert hasattr(self.host, self.attribute), f'Attribute {self.attribute} is not found.'
        
        if logs is None:
            logs = {}

        if (
            isinstance(self.save_freq, int)
            or self.epochs_since_last_save >= self.period
        ):
            self.epochs_since_last_save = 0

            try:
                filepath = self.add_suffix(path=self.filepath, epoch=epoch) if self.suffix is True else self.filepath
                
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        if self.verbose > 0:
                            warnings.warn(
                                f'Can save best model only with `{self.monitor}` '
                                'available, skipping'
                            )
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print(
                                    f'\nEpoch {epoch}: {self.monitor} improved '
                                    f'from {self.best:.4f} to {current:.4f}, '
                                    f'saving {self.attribute} to {filepath}'
                                )
                            self.best = current
                            torch.save(obj=getattr(self.host, self.attribute), f=filepath)
                        else:
                            if self.verbose > 0:
                                print(
                                    f"\nEpoch {epoch}: `{self.monitor}` did not"
                                    f" improve from {self.best:.4f}"
                                )
                else:
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch}: saving {self.attribute} to {filepath}')
                    torch.save(obj=getattr(self.host, self.attribute), f=filepath)

            except IsADirectoryError:
                raise IOError(
                    "Please specify a non-directory filepath for "
                    "ModelCheckpoint. Filepath used is an existing "
                    f"directory: {self.filepath}"
                )

    @staticmethod
    def add_suffix(path:str, epoch:int) -> str:
        path_parsed = Path(path)
        parent, name, extension = str(path_parsed.parent), path_parsed.stem, path_parsed.suffix
        
        path_new = f'{parent}/{name} - epoch {epoch}{extension}'
        return path_new