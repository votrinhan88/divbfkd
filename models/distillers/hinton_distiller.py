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


class HintonDistiller(Distiller):
    """Traditional knowledge distillation scheme, training the student on both
    actual data and the soft targets produced by a pre-trained teacher.

    Args:
    + `teacher`: Pre-trained teacher model. Must return logits.
    + `student`: To-be-trained student model. Must return logits.
    + `image_dim`: Dimension of input images. Defaults to `None`, skip to parse \
        from student.

    Kwargs:
    + `device`: The desired device of trainer, needs to be declared explicitly  \
        in case of Distributed Data Parallel training. Defaults to `None`, skip \
        to automatically choose single-process `'cuda'` if available or else    \
        `'cpu'`.
    + `world_size`: The number of processes in case of Distributed Data Parallel\
        training. Defaults to `None`, skips to query automatically with `torch`.
    + `master_rank`: Rank of the master process. Defaults to `0`.
    
    Distilling the Knowledge in a Neural Network - Hinton et al. (2015)
    DOI: 10.48550/arXiv.1503.02531
    """
    def compile(
        self,
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
        + `opt`: Optimizer for student model.
        + `loss_fn_distill`: Distillation loss function to match student's with \
            teacher's prediction. Pass in a custom function or toggle with      \
            `True`/`False`. Defaults to `True`.
        + `loss_fn_label`: Label loss function to match student's prediction    \
            with actual label. Pass in a custom function or toggle with `True`/ \
            `False`. Defaults to `True`.
        + `coeff_dt`: Multiplier assigned to distillation loss. Defaults to     \
            `0.9`.
        + `coeff_lb`: Multiplier assigned to label loss. Defaults to `0.1`.
        + `temperature`: Temperature for softening probability distributions.   \
            Larger temperature gives softer distributions. Defaults to `10`.

        Kwargs:
        + `loss_fn_test`: Test loss function to test student's prediction with  \
            ground truth label, of type `bool` or custom. `True`: use the cross-\
            entropy loss, `False`: skip measuring test loss, or pass in a custom\
            loss function. Defaults to `True`.
        + `sync_ddp_metrics`: Flag to synchronize metrics during Distributed    \
            Data Parallel training. Can be of type `bool` or dict of `bool`,    \
            accessible with the keys `'train'`|`'val'`. Defaults to `None` for  \
            off for training and on for validation. Warning: Turning this on    \
            will incur extra communication overhead.

        """
        super().compile(**kwargs)
        self.opt = opt
        self.loss_fn_distill = loss_fn_distill
        self.loss_fn_label = loss_fn_label
        self.coeff_dt = coeff_dt
        self.coeff_lb = coeff_lb
        self.temperature = temperature

        # Config distillation loss
        if self.loss_fn_distill is True:
            self._loss_fn_distill = nn.CrossEntropyLoss()
        elif self.loss_fn_distill is False:
            self._loss_fn_distill = lambda *args, **kwargs:torch.tensor(0)
        else:
            self._loss_fn_distill = self.loss_fn_distill
        # Config student loss
        if self.loss_fn_label is True:
            self._loss_fn_label = nn.CrossEntropyLoss()
        elif self.loss_fn_label is False:
            self._loss_fn_label = lambda *args, **kwargs:torch.tensor(0)
        else:
            self._loss_fn_label = self.loss_fn_label

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
        loss_distil = self._loss_fn_distill(input=soft_logits_S, target=soft_prob_T) * self.temperature**2
        # Label loss: standard loss with training data
        loss_label = self._loss_fn_label(logits_S, label)
        loss_S = (
            + self.coeff_dt*loss_distil
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
                self.train_metrics['loss_dt'].update(new_entry=loss_distil)
            if self.loss_fn_label is not False:
                self.train_metrics['loss_lb'].update(new_entry=loss_label)
            if ((self.loss_fn_distill is not False)
                | (self.loss_fn_label is not False)):
                self.train_metrics['loss_S'].update(new_entry=loss_S)
            self.train_metrics['acc'].update(prediction=logits_S, label=label)