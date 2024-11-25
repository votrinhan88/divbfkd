# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Any, Callable, Sequence

import torch
from torch import nn, Tensor

from models.distillers import Distiller
from utils.metrics import DDPMetric, Mean, CategoricalAccuracy
from utils.trainers import parse_loss


class HintonDistiller(Distiller):
    """Standard knowledge distillation scheme, training the student on both
    actual data and the soft targets produced by a pre-trained teacher.

    Kwargs: Additional arguments:
      + To `Distiller`: `teacher`, `student`, `image_dim`.
      + To `Trainer`: `device`, `world_size`, `master_rank`.
    
    Distilling the Knowledge in a Neural Network - Hinton et al. (2015).
    DOI: 10.48550/arXiv.1503.02531
    """
    def compile(self,
        opt:torch.optim.Optimizer,
        loss_fn_distill:bool|Callable[[Any], Tensor]=True,
        loss_fn_label:bool|Callable[[Any], Tensor]=True,
        coeff_dt:float=0.9,
        coeff_lb:float=0.1,
        temperature:float=10,
        **kwargs,
    ):
        """Compile distiller.

        Args:
          `opt`: Optimizer for student model.
          `loss_fn_distill`: Distillation loss function to match student's with \
            teacher's prediction. Pass in a custom function or toggle with      \
            `True`/`False`. Defaults to `True`.
          `loss_fn_label`: Label loss function to match student's prediction    \
            with actual label. Pass in a custom function or toggle with `True`/ \
            `False`. Defaults to `True`.
          `coeff_dt`: Multiplier assigned to distillation loss. Defaults to     \
            `0.9`.
          `coeff_lb`: Multiplier assigned to label loss. Defaults to `0.1`.
          `temperature`: Temperature for softening probability distributions.   \
            Larger temperature gives softer distributions. Defaults to `10`.
        
        Kwargs: Additional arguments:
          + To `Distiller().compile`: `loss_fn_test`.
          + To `Trainer().compile`: `sync_ddp_metrics`.
        """
        super().compile(**kwargs)
        self.opt = opt
        self.loss_fn_distill = loss_fn_distill
        self.loss_fn_label = loss_fn_label
        self.coeff_dt = coeff_dt
        self.coeff_lb = coeff_lb
        self.temperature = temperature

        # Config losses
        self._loss_fn_distill = parse_loss(loss_arg=loss_fn_distill, default=nn.CrossEntropyLoss())
        self._loss_fn_label = parse_loss(loss_arg=loss_fn_label, default=nn.CrossEntropyLoss())
        
        # Metrics
        if self.loss_fn_distill is not False:
            self.train_metrics.update({'loss_dt': Mean()})
        if self.loss_fn_label is not False:
            self.train_metrics.update({'loss_lb': Mean()})
        if ((self.loss_fn_distill is not False)
            | (self.loss_fn_label is not False)):
            self.train_metrics.update({'loss_S': Mean()})
        self.train_metrics.update({'acc': CategoricalAccuracy()})

        if self.ddp:
            for metric in ['loss_dt', 'loss_lb', 'loss_S', 'acc']:
                if self.sync_ddp_metrics['train']:
                    self.train_metrics.update({metric:DDPMetric(
                        metric=self.val_metrics.get(metric), 
                        device=self.device, 
                        world_size=self.world_size,
                    )})

    def train_batch(self, data:Sequence[Tensor]):
        # Unpack data
        input, label = data
        input, label = input.to(self.device), label.to(self.device)

        self.teacher.eval()
        self.student.train()
        self.opt.zero_grad()
        # Forward
        logits_T:Tensor = self.teacher(input)
        logits_S:Tensor = self.student(input)
        soft_prob_T = (logits_T/self.temperature).softmax(dim=1)
        soft_logits_S = (logits_S/self.temperature).log_softmax(dim=1)
        # Distillation loss: Not multiplying with T^2 gives slightly better results
        loss_distill = self._loss_fn_distill(input=soft_logits_S, target=soft_prob_T) * self.temperature**2
        # Label loss: standard loss with training data
        loss_label = self._loss_fn_label(logits_S, label)
        loss_S:Tensor = (
            + self.coeff_dt*loss_distill
            + self.coeff_lb*loss_label
        )
        
        # Backward
        if ((self.loss_fn_distill is not False)
            | (self.loss_fn_label is not False)):
            loss_S.backward()
            self.opt.step()
        
        with torch.inference_mode():
            # Metrics
            if self.loss_fn_distill is not False:
                self.train_metrics['loss_dt'].update(new_entry=loss_distill)
            if self.loss_fn_label is not False:
                self.train_metrics['loss_lb'].update(new_entry=loss_label)
            if ((self.loss_fn_distill is not False)
                | (self.loss_fn_label is not False)):
                self.train_metrics['loss_S'].update(new_entry=loss_S)
            self.train_metrics['acc'].update(prediction=logits_S, label=label)