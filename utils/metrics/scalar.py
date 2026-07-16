from typing import Any, Optional

import torch

from .metric import Metric


class Scalar(Metric):
    """Keep track of a scalar metric. Can be updated by overwriting it.

    Args:
    + `persistent`: Flag to set metric persistent, i.e., not resetting to `NaN` \
        before each epoch. Defaults to `True`.
    + `ddp_src_device`: Source device to update in case of Distributed Data     \
        Parallel. Defaults to `None`, allowing updates from all devices in   \
        an epoch.
    """
    def __init__(self,
        persistent:bool=True,
        ddp_src_device:Optional[torch.device]=None,
    ):
        assert isinstance(persistent, bool), '`persistent` must be of type bool.'
        self.persistent = persistent
        self.ddp_src_device = ddp_src_device

        self.hard_reset()

    def update(self, new_entry:int|float):
        self.scalar = new_entry

    def update_ddp(self,
        new_entry:int|float,
        device:torch.device=0,
        world_size:int=1,
        collect_kwargs:dict[str, Any]={},
    ):
        if (self.ddp_src_device is None) | (device == self.ddp_src_device):
            self.update(new_entry=new_entry)

    def reset(self):
        if self.persistent is False:
            self.hard_reset()
    
    def hard_reset(self):
        self.scalar:int|float = float('nan')
    
    @property
    def value(self):
        return self.scalar

    @property
    def py_value(self):
        return self.value