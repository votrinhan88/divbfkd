import torch
from torch import nn, Tensor


class OneHotEncoding(nn.Module):
    """Apply one-hot encoding.

    Args:
    + `num_classes`: Number of classes.
    + `dtype`: Dtype to return. Defaults to `torch.float`.
    """        
    def __init__(self, num_classes:int, dtype:torch.dtype=torch.float):
        super().__init__()
        self.num_classes = num_classes
        self.dtype = dtype
    
    def forward(self, input:Tensor) -> Tensor:
        x = nn.functional.one_hot(input.long(), num_classes=self.num_classes)
        x = x.squeeze(dim=-2)
        x = x.to(dtype=self.dtype)
        return x

    def extra_repr(self) -> str:
        return f'num_classes={self.num_classes}, dtype={self.dtype}'