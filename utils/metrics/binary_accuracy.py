from typing import Any

import torch
from torch import Tensor
import torch.distributed as dist

from .metric import Metric


class BinaryAccuracy(Metric):
    """Binary accuracy, typically used for binary classification problems.

    Args:
    + `threshold`: Threshold to decide positive examples (1 when prediction >   \
        threshold and 0 vice versa). Defaults to `0.5`.
    """
    def __init__(self, threshold:float=0.5):
        self.threshold = threshold
        self.reset()
    
    def update(self, prediction:Tensor, label:Tensor):
        # `prediction` should be probability
        prediction = prediction.to('cpu')
        label = label.to('cpu')

        pred_label = (prediction >= self.threshold).to(dtype=torch.int)
        self.num_observed += label.shape[0]

        self.label = torch.cat(tensors=[self.label, label], dim=0)
        self.pred_label = torch.cat(tensors=[self.pred_label, pred_label], dim=0)

    def update_ddp(self,
        prediction:Tensor,
        label:Tensor,
        device:torch.device=0,
        world_size:int=1,
        collect_kwargs:dict[str, Any]={},
    ):
        list_prediction = [torch.zeros_like(prediction) for i in range(world_size)]
        dist.all_gather(tensor_list=list_prediction, tensor=prediction)

        list_label = [torch.zeros_like(label) for i in range(world_size)]
        dist.all_gather(tensor_list=list_label, tensor=label)

        prediction = torch.cat(tensors=[i for i in list_prediction], dim=0)
        label = torch.cat(tensors=[i for i in list_label], dim=0)
        self.update(prediction=prediction, label=label)

    def reset(self):
        self.label = torch.empty(size=[0], dtype=torch.int)
        self.pred_label = torch.empty(size=[0], dtype=torch.int)
        self.num_observed = torch.tensor(0, dtype=torch.long)

    @property
    def value(self):
        return (self.label == self.pred_label).sum()/self.num_observed