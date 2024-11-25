from typing import Any, Optional, Sequence

from torch import device, Tensor

from .metric import Metric


class DDPMetric(Metric):
    """Wrapper for metrics to support Distributed Data Parallel logic.

    Args:
    + `metric`: The metric to use in DDP.
    + `device`: The current device wrapped with `torch.device`.
    + `world_size`: World size of DDP training configuration.
    + `collect_kwargs`: Additional keyword arguments to pass into the gathering \
        operators in `metric.update_ddp()`. Defaults to `{'dst':0}`.
    """
    def __init__(
        self,
        metric:Metric,
        device:device,
        world_size:int,
        collect_kwargs:Optional[dict[str, Any]]=None,
    ):
        super().__init__()
        if not isinstance(metric, Metric):
            raise TypeError(f'Metric {metric} is of inappropriate type.')
        self.metric = metric
        self.device = device
        self.world_size = world_size
        self.collect_kwargs = collect_kwargs

        if self.collect_kwargs is None:
            self.collect_kwargs = {}

    def update(self, *args:Sequence[Tensor], **kwargs:Tensor|Sequence[Tensor]):
        self.metric.update_ddp(
            *args, **kwargs,
            device=self.device,
            world_size=self.world_size,
            collect_kwargs=self.collect_kwargs,
        )

    def reset(self):
        return self.metric.reset()
    
    @property
    def value(self):
        return self.metric.value

    @property
    def py_value(self):
        return self.metric.py_value
    
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.metric)})"