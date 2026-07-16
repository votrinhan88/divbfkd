from typing import Any
from functools import partial

from numpy import ndarray
import torch
from torch import Tensor
import torch.distributed as dist

from .metric import Metric


class MultiCounter(Metric):
    """Counter that works for multiple classes.

    Args:
    + `num_classes`: Number of classes.
    + `reduction`: Reduction applying to the new entries: `'none'` | `'argmax'` \
        | `'argmin'`. `'none'`: the new entries is not reduced, typically used  \
        for already reduced probability distributions, `'argmax'`: reduce by    \
        argmax to aggregate, `'argmin'`: reduce by argmin to aggregate. Defaults\
        to `'none'`.
    + `cumulate`: Flag to count cumulatively over every epochs. Defaults to     \
        `False`.
    + `dtype`: Dtype of the counting values. Defaults to `torch.long`.

    Use cases:
    ``Notes``
    + Counting the predictions of a classifer from logits for each epoch:       \
        reduction = 'argmax', cumulate = False, dtype = torch.long
    + Counting the number of samples class-wise that has been used to train a   \
        neural network from beginning to end, from class labels: reduction =    \
        'none', cumulate = True, dtype = torch.long
    """
    def __init__(
        self,
        num_classes:int,
        reduction:str='none',
        cumulate:bool=False,
        dtype:torch.dtype=torch.long,
    ):
        assert isinstance(cumulate, bool), '`cumulate` must be of type bool.'
        assert reduction in ['none', 'argmax', 'argmin'], "`reduction` must be one of 'none', 'argmax', 'argmin'."
        self.num_classes = num_classes
        self.reduction = reduction
        self.cumulate = cumulate
        self.dtype = dtype

        if self.reduction is None:
            self._reduction = lambda x:x
        elif self.reduction == 'argmax':
            self._reduction = partial(torch.argmax, dim=1)
        elif self.reduction == 'argmin':
            self._reduction = partial(torch.argmin, dim=1)

        self.hard_reset()
        
    def update(self, new_entry:Tensor):
        new_entry = new_entry.detach().to(device='cpu')

        self.step += 1
        new_count = self._reduction(new_entry)
        self.count += torch.bincount(input=new_count, minlength=self.num_classes)

    def update_ddp(self,
        new_entry:Tensor,
        device:torch.device=0,
        world_size:int=1,
        collect_kwargs:dict[str, Any]={},
    ):
        list_entry = [torch.zeros_like(new_entry) for i in range(world_size)]
        dist.all_gather(tensor_list=list_entry, tensor=new_entry)
        new_entry = torch.cat(tensors=[i for i in list_entry], dim=0)
        self.update(new_entry=new_entry)

    def reset(self):
        if self.cumulate is False:
            self.hard_reset()
    
    def hard_reset(self):
        self.step = 0
        self.count = torch.zeros(size=[self.num_classes], dtype=self.dtype)

    @property
    def value(self) -> ndarray:
        return self.count.numpy()

    @property
    def py_value(self) -> ndarray:
        return self.value
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(py_value={self.py_value})"