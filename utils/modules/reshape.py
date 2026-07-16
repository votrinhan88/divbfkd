from typing import Sequence
from torch import nn, Tensor


class Reshape(nn.Module):
    """Reshape the input to the given shape.

    Args:
    + `out_shape`: Output shape.
    """
    def __init__(self, out_shape:Sequence[int]):
        super().__init__()
        self.out_shape = out_shape
    
    def forward(self, input:Tensor) -> Tensor:
        batch_size = input.shape[0]
        x = input.view([batch_size, *self.out_shape])
        return x
    
    def extra_repr(self) -> str:
        return f'out_shape={self.out_shape}'