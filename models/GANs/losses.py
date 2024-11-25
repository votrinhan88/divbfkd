# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

import torch
from torch import nn, Tensor

from utils.modules import get_reduction_fn


class WassersteinLoss(nn.Module):
    """Wasserstein loss function, also known as Earth Mover's Distance loss.
    
    Args:
    + `reduction`: Reduction applying to the output: `'mean'` | `'sum'` |       \
        `'none'`. Defaults to `'mean'`.
    """
    def __init__(self, reduction:str='mean'):
        super().__init__()
        self.reduction = reduction
        
        self.reduction_fn = get_reduction_fn(reduction=reduction)

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        loss = input - target
        loss = self.reduction_fn(loss)
        return loss
    
    def extra_repr(self) -> str:
        params = {'reduction': self.reduction}
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class SoftplusLoss(nn.Module):
    """Softplus loss function, operates with different signs depending on real/
    fake labels and generator/discriminator.
    
    Args:
    + `reduction`: Reduction applying to the output: `'mean'` | `'sum'` |       \
        `'none'`. Defaults to `'mean'`.
    """
    sign_model = {'d': 1, 'g':-1}
    sign_authenticity = {'real': -1, 'fake': 1}

    def __init__(self, reduction:str='mean', **kwargs):
        super().__init__()
        self.reduction = reduction
        
        self.reduction_fn = get_reduction_fn(reduction=reduction)
        self.softplus = nn.Softplus(**kwargs)

    def forward(self, input:Tensor, sign:float|str) -> Tensor:
        sign = self.parse_sign(sign)
        loss = self.softplus(sign*input)
        loss = self.reduction_fn(loss)
        return loss
    
    def parse_sign(self, sign:float|str) -> float:
        if not isinstance(sign, (int, float)):
            assert isinstance(sign, str)
            model, authenticity = sign.split('-')
            sign = self.sign_model[model]*self.sign_authenticity[authenticity]
        return sign

    def extra_repr(self) -> str:
        params = {'reduction': self.reduction}
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class R1RegularizationLoss(nn.Module):
    """R1 regularization for GAN discriminator.
    https://github.com/ChristophReich1996/Dirac-GAN/blob/main/dirac_gan/loss.py#L292
    
    Args:
    + `reduction`: Reduction applying to the output: `'mean'` | `'sum'` |       \
        `'none'`. Defaults to `'mean'`.
    
    Which Training Methods for GANs do actually Converge? - Mescheder et al.
    """
    def __init__(self, reduction:str='mean'):
        super().__init__()
        self.reduction = reduction
        self.reduction_fn = get_reduction_fn(reduction=reduction)

    def forward(self, discriminator:nn.Module, x_real:Tensor, **kwargs) -> Tensor:
        # Detach to build a separate graph
        x_real = x_real.clone().detach()
        x_real.requires_grad = True
        
        # Compute gradient
        pred_real = discriminator(x_real, **kwargs)
        grad = torch.autograd.grad(
            inputs=x_real,
            outputs=pred_real, 
            grad_outputs=torch.ones_like(pred_real),
            create_graph=True,
        )[0]
        
        # Compute loss
        loss = grad.flatten(start_dim=1, end_dim=-1).pow(exponent=2).sum(dim=1)
        loss = self.reduction_fn(loss)
        return loss


class GradientPenalty1LipschitzLoss(nn.Module):
    """Gradient penalty to constrain 1-Lipschitz continuity to the discriminator
    as introduced in WGAN-GP.

    Args:
    + `reduction`: Reduction applying to the output: `'mean'` | `'sum'` |       \
    `'mean_sum'` | `'none'`. Defaults to `'mean'`.
    
    Advantages:
    + `x_real` and `x_synth` are properly detached from upstream graph. Instead,\
        `interpolated.requires_grad` is set manually.
    + The flag `retain_graph=True` is skipped for flexibility at a small price  \
        of efficiency. Now gradient penalty no longer needs to be computed other\
        losses.
    + Input args to the discriminator is flexible with the use of kwargs (e.g.  \
        step and alpha as in ProGAN).
    + Modular and reusable.
    """
    def __init__(self, reduction:str='mean'):
        super().__init__()
        self.reduction = reduction
        self.reduction_fn = get_reduction_fn(reduction=reduction)

    def forward(self, discriminator:nn.Module, x_real:Tensor, x_synth:Tensor, **kwargs) -> Tensor:
        # Detach to build a separate graph
        x_real = x_real.clone().detach()
        x_synth = x_synth.clone().detach()
        
        # Create interpolated images of real and synthetic images
        image_dim = x_real.shape[1:]
        epsilon = torch.randn(
            size=[x_real.shape[0], *[1]*len(image_dim)],
            device=x_real.device,
        )
        epsilon = epsilon.repeat(1, *image_dim)
        interpolated = x_real*epsilon + x_synth*(1 - epsilon)
        interpolated.requires_grad = True 
        
        # Compute gradient
        mixed_scores = discriminator(interpolated, **kwargs)
        grad = torch.autograd.grad(
            inputs=interpolated,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
        )[0]

        # Compute loss
        loss = (grad.flatten(start_dim=1, end_dim=-1).norm(2, dim=1) - 1)**2
        loss = self.reduction_fn(loss)
        return loss