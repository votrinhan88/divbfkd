from typing import Callable, Optional, Sequence

from torch import nn, Tensor
from torch.nn.modules.loss import _Loss


class CompositeLoss(nn.Module):
    """Combines mutiple loss functions.

    Args:
    + `loss_fn`: A sequence of loss functions to combine.
    + `weights`: Weights for the loss functions with the same length. Defaults  \
        to `None`, skip to set equally divided weights.
    """
    def __init__(self,
        loss_fn:Sequence[_Loss],
        weights:Optional[Sequence[float]]=None,
    ):
        if weights is not None:
            assert len(weights) == len(loss_fn)
        
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights
        
        if self.weights is None:
            self.weights = [1/len(loss_fn)] * len(loss_fn)

    def forward(self, *args, **kwargs):
        losses = [w*l_fn(*args, **kwargs)
                    for w, l_fn in zip(self.weights, self.loss_fn)]
        return sum(losses)


class CustomLossWrapper(_Loss):
    """Wraps a loss function and allows more flexible reduction.

        Args:
        + `loss_fn`: Loss function to wrap, must have `reduction='none'`.
        + `reduction`: Reduction method to retrieve with `get_reduction_fn`.    \
            Defaults to `'mean'`.
    """
    def __init__(self, loss_fn:_Loss, reduction:str='mean'):
        if loss_fn.reduction != 'none':
            raise ValueError("`loss_fn` must have `reduction`='none'.")
        
        super().__init__()
        self.loss_fn = loss_fn
        self.reduction = reduction

        self.reduction_fn = get_reduction_fn(reduction=reduction)

    def forward(self, *args, **kwargs) -> Tensor:
        loss = self.loss_fn(*args, **kwargs)
        loss = self.reduction_fn(loss)
        return loss

    def extra_repr(self) -> str:
        return f'reduction={self.reduction}'


def get_reduction_fn(reduction:Optional[str]='mean') -> Callable[[Tensor], Tensor]:
    """Get reduction function.

    Args:
    + `reduction`: Reduction applying to the output: `'mean'` | `'sum'` |       \
        `'mean_sum'` | `'none'`. `'mean'`: the mean of the output is taken,     \
        `'sum'`: the output will be summed, `'mean_sum'`: the output will be    \
        summed until it is one-dimensional (the batch dimension) then the mean  \
        is taken, `'none'`: no reduction will be applied. Defaults to `'mean'`.
    """
    assert reduction in ['mean', 'sum', 'mean_sum', 'none'], (
        f"'{reduction}' is not supported."
    )
    
    if reduction == 'mean':
        return lambda x:x.mean(dim=0)
    elif reduction == 'sum':
        return lambda x:x.sum(dim=0)
    elif reduction == 'mean_sum':
        def mean_sum(input:Tensor) -> Tensor:
            while input.dim() >= 2:
                input = input.sum(dim=-1)
            input = input.mean(dim=0)
            return input
        return mean_sum
    elif reduction == 'none':
        return lambda x:x
    else:
        raise ValueError(f"'{reduction}' is not available.")