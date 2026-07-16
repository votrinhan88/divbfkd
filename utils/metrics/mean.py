from typing import Any

import torch
from torch import Tensor
import torch.distributed as dist

from .metric import Metric


class Mean(Metric):
    """Mean of the values for every batches in an epoch."""
    def __init__(self):
        self.reset()

    def update(self, new_entry:Tensor):
        new_entry = new_entry.to('cpu')

        self.step += 1
        self.accum_value += new_entry

    def update_ddp(self,
        new_entry:Tensor,
        device:torch.device,
        world_size:int=1,
        collect_kwargs:dict[str, Any]={},
    ):
        new_entry = new_entry.to(device=device)
        dist.all_reduce(tensor=new_entry, op=dist.ReduceOp.SUM, **collect_kwargs)
        new_entry = new_entry/world_size
        self.update(new_entry=new_entry)

    def reset(self):
        self.step:int = 0
        self.accum_value = torch.zeros(1)

    @property
    def value(self):
        return self.accum_value/self.step