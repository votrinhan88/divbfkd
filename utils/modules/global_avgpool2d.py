from torch import nn, Tensor


class GlobalAvgPool2d(nn.Module):
    """Apply global average pooling that take the average of each feature map.
    Only works for NCHW or CHW tensors.

    Args:
    + `keepdim`: Flag to retain the H and W dimensions in the output tensor.    \
        Defaults to `False`. 
    """ 
    def __init__(self, keepdim:bool=False):
        super().__init__()
        self.keepdim = keepdim

    def forward(self, input:Tensor) -> Tensor:
        x = input.mean(dim=[-1, -2], keepdim=self.keepdim)
        return x

    def extra_repr(self) -> str:
        return f'keepdim={self.keepdim}'