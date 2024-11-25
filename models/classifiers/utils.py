# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'divbfkd', "Wrong parent folder. Please change to 'divbfkd'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.trainers import Trainer
from utils.metrics import CategoricalAccuracy, DDPMetric, Mean
from utils.trainers import parse_loss


class ClassifierTrainer(Trainer):    
    """Trainer framework to train classifiers.
        
    Args:
      `model`: Classifier model to train. Must return logits.

    Kwargs: Additional arguments to `Trainer`: `device`, `world_size`,
    `master_rank`
    """
    def __init__(self, model:nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.model = model.to(device=self.device)
        if self.ddp:
            self.model = DDP(module=model, device_ids=[self.rank])

    def compile(self,
        opt:Optional[torch.optim.Optimizer]=None,
        loss_fn:bool|Callable[[Any], Tensor]=True,
        **kwargs,
    ):
        """Compile trainer.

        Args:
          `opt`: Optimizer. Defaults to `None`, skip to not use an optimizer    \
            (model is not updated).
          `loss_fn`: Loss function of type `bool` or custom. `True`: use the    \
            cross-entropy loss, `False`: do not use any losses (model is not    \
            updated), or pass in a custom loss function. Defaults to `True`.
        
        Kwargs: Additional arguments to `Trainer().compile`: `sync_ddp_metrics`.
        """
        super().compile(**kwargs)
        self.opt = opt
        self.loss_fn = loss_fn
        
        # Config loss function
        self._loss_fn = parse_loss(loss_arg=loss_fn, default=nn.CrossEntropyLoss())
        
        # Metrics
        self.train_metrics.update({'acc': CategoricalAccuracy()})
        self.val_metrics.update({'acc': CategoricalAccuracy()})
        if self.loss_fn is not False:
            self.train_metrics.update({'loss': Mean()})
            self.val_metrics.update({'loss': Mean()})
        # Wrap metrics in DDP
        if self.ddp:
            for metric in ['loss', 'acc']:
                if self.sync_ddp_metrics['train']:
                    self.train_metrics.update({metric:DDPMetric(
                        metric=self.train_metrics.get(metric), 
                        device=self.device, 
                        world_size=self.world_size,
                    )})
                if self.sync_ddp_metrics['val']:
                    self.val_metrics.update({metric:DDPMetric(
                        metric=self.val_metrics.get(metric), 
                        device=self.device, 
                        world_size=self.world_size,
                    )})

    def train_batch(self, data:Tuple[Tensor, Tensor]):
        # Unpack data
        input, label = data
        input, label = input.to(device=self.device), label.to(device=self.device)
        
        self.model.train()
        self.opt.zero_grad()
        # Forward
        prediction = self.model(input)
        loss:Tensor = self._loss_fn(input=prediction, target=label)
        # Backward
        if self.loss_fn is not False:
            loss.backward()
            self.opt.step()
        with torch.inference_mode():
            # Metrics
            if self.loss_fn is not False:
                self.train_metrics['loss'].update(new_entry=loss)
            self.train_metrics['acc'].update(label=label, prediction=prediction)

    def test_batch(self, data:Tuple[Tensor, Tensor]):
        # Unpack data
        input, label = data
        input, label = input.to(device=self.device), label.to(device=self.device)
        
        self.model.eval()
        with torch.inference_mode():
            # Forward
            prediction = self.model(input)
            loss = self._loss_fn(input=prediction, target=label)
            # Metrics
            if self.loss_fn is not False:
                self.val_metrics['loss'].update(new_entry=loss)
            self.val_metrics['acc'].update(label=label, prediction=prediction)