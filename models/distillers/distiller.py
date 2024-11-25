# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Any, Callable, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.metrics import DDPMetric, Mean, CategoricalAccuracy
from utils.trainers import Trainer


class Distiller(Trainer):
    """Base class for knowledge distillation models.
    
    Args:
    + `teacher`: Pre-trained teacher model.
    + `student`: To-be-trained student model.
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
    """
    def __init__(self,
        teacher:nn.Module,
        student:nn.Module,
        image_dim:Optional[Sequence[int]]=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher = teacher.to(device=self.device)
        # Teacher is frozen for inference only
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.student = student.to(device=self.device)
        self.image_dim = image_dim

        if self.image_dim is None:
            self.image_dim:int = self.student.input_dim

        if self.ddp:
            self.student = DDP(module=student, device_ids=[self.rank])

    def compile(self,
        loss_fn_test:bool|Callable[[Any], Tensor]=True,
        **kwargs,
    ):
        """Compile distiller.

        Args:
        + `loss_fn_test`: Test loss function to test student's prediction with  \
            ground truth label, of type `bool` or custom. `True`: use the cross-\
            entropy loss, `False`: skip measuring test loss, or pass in a custom\
            loss function. Defaults to `True`.
        
        Kwargs:
        + `sync_ddp_metrics`: Flag to synchronize metrics during Distributed    \
            Data Parallel training. Can be of type `bool` or dict of `bool`,    \
            accessible with the keys `'train'`|`'val'`. Defaults to `None` for  \
            off for training and on for validation. Warning: Turning this on    \
            will incur extra communication overhead.
        """
        super().compile(**kwargs)
        self.loss_fn_test = loss_fn_test

        # Config test loss
        if self.loss_fn_test is True:
            self._loss_fn_test = nn.CrossEntropyLoss()
        elif self.loss_fn_test is False:
            self._loss_fn_test = lambda *args, **kwargs:torch.tensor(0)
        else:
            self._loss_fn_test = self.loss_fn_test

        # Metrics
        self.val_metrics = {
            'loss': Mean(),
            'acc': CategoricalAccuracy(),
        }
        # Wrap metrics in DDP
        if self.ddp:
            for metric in ['loss', 'acc']:
                if self.sync_ddp_metrics['val']:
                    self.val_metrics.update({metric:DDPMetric(
                        metric=self.val_metrics.get(metric), 
                        device=self.device, 
                        world_size=self.world_size,
                    )})

    def train_batch(self, data:Sequence[Tensor]):
        raise NotImplementedError('Subclasses should be implemented first.')

    def test_batch(self, data:Sequence[Tensor]):
        # Unpack data
        input, label = data
        input, label = input.to(device=self.device), label.to(device=self.device)

        self.student.eval()
        with torch.inference_mode():
            # Forward
            logits_S = self.student(input)
            loss_S = self._loss_fn_test(logits_S, label)
            # Metrics
            if self.loss_fn_test is not False:
                self.val_metrics['loss'].update(new_entry=loss_S)
            self.val_metrics['acc'].update(prediction=logits_S, label=label)