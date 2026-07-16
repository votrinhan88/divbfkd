from typing import Any, Callable
from torch import nn, tensor, Tensor

def parse_loss(
    loss_arg:bool|Callable[[Any], Tensor],
    default:Callable[[Any], Tensor],
) -> Callable[[Any], Tensor]:
    if loss_arg is True:
        return default
    elif loss_arg is False:
        return lambda *args, **kwargs:tensor(0)
    else:
        return loss_arg

def parse_module(
    arg:bool|Callable[[Any], Tensor],
    default_true:Callable[[Any], Tensor],
    default_false:Callable[[Any], Tensor]=nn.Identity(),
) -> Callable[[Any], Tensor]:
    if arg is True:
        return default_true
    elif arg is False:
        return default_false
    else:
        return arg