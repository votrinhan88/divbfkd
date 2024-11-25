from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
from torch import device, Tensor
from torch.utils.data import DataLoader
import tqdm.auto as tqdm

from utils.metrics import Metric
from utils.callbacks import (
    Callback,
    History,
    Iterator,
    ProgressBar,
    SamplerUpdater
)


class Trainer(ABC):
    """Base class for trainers.

    Args:
    + `device`: The desired device of trainer, needs to be declared explicitly  \
        in case of Distributed Data Parallel training. Defaults to `None`, skip \
        to automatically choose single-process `'cuda'` if available or else    \
        `'cpu'`.
    + `world_size`: The number of processes in case of Distributed Data Parallel\
        training. Defaults to `None`, skips to query automatically with `torch`.
    + `master_rank`: Rank of the master process. Defaults to `0`.
    """
    @abstractmethod
    def __init__(self,
        device:Optional[str|device]=None,
        world_size:Optional[int]=None,
        master_rank:int=0,
    ):
        # Parse device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Parse world_size
        if world_size == None:
            world_size = torch.cuda.device_count()
        self.world_size = world_size

        self.master_rank = master_rank

        # Auto select rank and DDP
        self.rank = self.device.index
        self.ddp = (self.world_size > 1) and (self.rank != None)
        if self.ddp:
            self.master_rank = master_rank

    @abstractmethod
    def compile(self, sync_ddp_metrics:Optional[bool|dict[str, bool]]=None):
        """Compile base trainer and define necessary attributes.
        
        Args:
        + `sync_ddp_metrics`: Flag to synchronize metrics during Distributed    \
            Data Parallel training. Can be of type `bool` or dict of `bool`,    \
            accessible with the keys `'train'`|`'val'`. Defaults to `None` for  \
            off for training and on for validation. Warning: Turning this on    \
            will incur extra communication overhead.
        """
        if sync_ddp_metrics is None:
            self.sync_ddp_metrics = {'train':False, 'val':True}
        elif isinstance(sync_ddp_metrics, bool):
            self.sync_ddp_metrics = {
                s:sync_ddp_metrics for s in ['train', 'val']
            }
        else:
            self.sync_ddp_metrics = sync_ddp_metrics

        # To be initialize by callback `History`
        self.history:History
        # To be initialize by callback `ProgressBar`
        self.training_progress:tqdm    = None
        self.train_phase_progress:tqdm = None
        self.val_phase_progress:tqdm   = None
        # Metrics
        self.train_metrics:dict[str, Metric] = {}
        self.val_metrics:dict[str, Metric]   = {}

    def training_loop(self,
        trainloader:DataLoader,
        num_epochs:int,
        valloader:Optional[DataLoader]=None,
        callbacks:Optional[Sequence[Callback]]=None,
        start_epoch:int=0,
        val_freq:int=1,
    ) -> History:
        """Base training loop.
        
        Args:
        + `trainloader`: Dataloader for training set.
        + `num_epochs`: Number of epochs.
        + `valloader`: Dataloader for validation set. Defaults to `None`.
        + `callbacks`: List of callbacks. Defaults to `None`.
        + `start_epoch`: Index of first epoch. Defaults to `0`.
        + `val_freq`: Number of training epoch to perform a validation step.   \
            Defaults to `1`.
        """
        self.training_loop_kwargs = {
            'trainloader':trainloader,
            'num_epochs':num_epochs,
            'valloader':valloader,
            'start_epoch':start_epoch,
        }

        self.hook_callbacks(callbacks=callbacks)
        logs = {}
        self.on_train_begin(logs)
        
        epoch_logs = {}
        for epoch in self.training_progress:
            epoch_logs.clear()
            self.on_epoch_begin(epoch, epoch_logs)
            
            # Training phase
            for batch, data in enumerate(self.train_phase_progress):
                batch_logs = {}
                self.on_train_batch_begin(batch, batch_logs)
                self.train_batch(data)
                batch_logs.update({k:v.py_value for (k, v) in self.train_metrics.items()})
                self.on_train_batch_end(batch, batch_logs)
            epoch_logs.update(batch_logs)
            self.on_epoch_train_end(epoch, epoch_logs)
            
            # Validation phase
            if (valloader is not None) & ((epoch - start_epoch) % val_freq == 0):
                self.on_epoch_test_begin(epoch, epoch_logs)
                for batch, data in enumerate(self.val_phase_progress):
                    batch_logs = {} 
                    self.on_test_batch_begin(batch, batch_logs)
                    self.test_batch(data)
                    batch_logs.update({k:v.py_value for (k, v) in self.val_metrics.items()})
                    self.on_test_batch_end(batch, batch_logs)
                epoch_logs.update({f'val_{key}': value for key, value in batch_logs.items()})
            self.on_epoch_end(epoch, epoch_logs)
        
        logs.update(epoch_logs)
        self.on_train_end(logs)

        self.training_loop_kwargs.clear()

        return self.history

    def evaluate(self,
        valloader:DataLoader,
        callbacks:Optional[Sequence[Callback]]=None,
    ) -> History:
        """Base evaluation step.
        
        Args:
        + `valloader`: Dataloader for validation/test set.
        + `callbacks`: List of callbacks. Defaults to `None`.
        """        
        self.evaluate_kwargs = {'valloader':valloader}

        self.hook_callbacks(callbacks=callbacks)
        logs = {}
        self.on_test_begin(logs)

        # Validation phase
        epoch_logs = {}
        for batch, data in enumerate(self.val_phase_progress):
            batch_logs = {}
            self.on_test_batch_begin(batch, batch_logs)
            self.test_batch(data)
            batch_logs.update({k:v.py_value for (k, v) in self.val_metrics.items()})
            self.on_test_batch_end(batch, batch_logs)
        epoch_logs.update({f'val_{key}': value for key, value in batch_logs.items()})
        
        logs.update(epoch_logs)
        self.on_test_end(logs)
        
        self.evaluate_kwargs.clear()

        return self.history

    @abstractmethod
    def train_batch(self, data:Sequence[Tensor]):
        """Train step for a batch.
        
        Args:
        + `data`: Training data from a dataloader.
        """
        raise NotImplementedError('Subclasses should be implemented first.')

    @abstractmethod
    def test_batch(self, data:Sequence[Tensor]):
        """Test step for a batch.
        
        Args:
        + `data`: Test data from a dataloader.
        """
        raise NotImplementedError('Subclasses should be implemented first.')

    def hook_callbacks(self, callbacks:Optional[Sequence[Callback]]):
        """Hook callbacks to trainer, always include a `History()` and a
        `ProgressBar()` first.
        
        Args:
        + `callbacks`: List of callbacks.
        """
        self.callbacks:Sequence[Callback] = [History()]

        if (not self.ddp) or (self.ddp and self.rank == self.master_rank):
            self.callbacks.append(ProgressBar())
        else:
            self.callbacks.append(Iterator())

        if self.ddp:
            self.callbacks.append(SamplerUpdater())

        if callbacks is not None:
            self.callbacks.extend(callbacks)
        for cb in self.callbacks:
            cb.hook(self)

    def on_train_begin(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_test_begin(self, logs=None):
        for cb in self.callbacks:
            cb.on_test_begin(logs)
        # Reset metrics
        for metric in [*self.val_metrics.values()]:
            metric.reset()

    def on_test_end(self, logs=None):
        for cb in self.callbacks:
            cb.on_test_end(logs)

    def on_epoch_begin(self, epoch:int, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)
        # Reset metrics
        for metric in [*self.train_metrics.values(), *self.val_metrics.values()]:
            metric.reset()
    
    def on_epoch_train_end(self, epoch:int, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_train_end(epoch, logs)
    
    def on_epoch_test_begin(self, epoch:int, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_test_begin(epoch, logs)

    def on_epoch_end(self, epoch:int, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch:int, logs=None):
        for cb in self.callbacks:
            cb.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch:int, logs=None):
        for cb in self.callbacks:
            cb.on_train_batch_end(batch, logs)

    def on_test_batch_begin(self, batch:int, logs=None):
        for cb in self.callbacks:
            cb.on_test_batch_begin(batch, logs)

    def on_test_batch_end(self, batch:int, logs=None):
        for cb in self.callbacks:
            cb.on_test_batch_end(batch, logs)

    @property
    def name(self) -> str:
        return f'{self.__class__.__name__}'