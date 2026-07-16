from typing import Any

import torch
from torch import Tensor
import torch.distributed as dist

from .metric import Metric


class Counter(Metric):
    """Counter that cumulates a scalar with every batches.

    Args:
    + `dtype`: Init dtype of the sum. Defaults to `torch.long`.
    + `cumulate`: Flag to cumulate the sum with every epochs, i.e., the whole   \
        training process. Defaults to `False`.
    """
    def __init__(self, dtype=torch.long, cumulate:bool=False):
        assert isinstance(cumulate, bool), '`cumulate` must be of type bool.'
        self.dtype = dtype
        self.cumulate = cumulate

        self.hard_reset()

    def update(self, new_entry:int|float):
        self.count += new_entry

    def update_ddp(self,
        new_entry:int|float,
        device:torch.device=0,
        world_size:int=1,
        collect_kwargs:dict[str, Any]={},
    ):
        new_entry:Tensor = torch.tensor(new_entry, device=device)
        dist.all_reduce(tensor=new_entry, op=dist.ReduceOp.SUM, **collect_kwargs)
        self.update(new_entry=new_entry.item())

    def reset(self):
        if self.cumulate is False:
            self.hard_reset()
    
    def hard_reset(self):
        self.count = torch.tensor(0, dtype=self.dtype)
    
    @property
    def value(self):
        return self.count

    @property
    def py_value(self):
        return self.value.item()