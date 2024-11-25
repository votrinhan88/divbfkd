# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch import nn, Tensor

from models.GANs import _GAN
from models.GANs.losses import WassersteinLoss, GradientPenalty1LipschitzLoss
from utils.metrics import Mean, BinaryAccuracy
from utils.trainers import parse_loss


class Discriminator(nn.Module):
    """Discriminator from DCGAN but with instance norm."""
    norm_dict = {'batchnorm': nn.BatchNorm2d, 'instancenorm':nn.InstanceNorm2d}
    
    def __init__(self,
        image_dim:Sequence[int]=[1, 28, 28],
        base_dim:Sequence[int]=[256, 7, 7],
        return_logits:bool=True,
        norm:Optional[str]='instancenorm',
    ):
        # Parse architecture from input dimension
        dim_ratio = [image_dim[axis]/base_dim[axis] for axis in torch.arange(start=1, end=len(image_dim))]
        for axis in range(len(dim_ratio)):
            num_conv = np.log(dim_ratio[axis])/np.log(2.)
            assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
            assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
        num_conv = int(num_conv)
        
        super().__init__()
        self.image_dim = image_dim
        self.base_dim = base_dim
        self.return_logits = return_logits
        self.norm = norm

        conv_blocks = [None for i in range(num_conv)]
        in_channels = self.image_dim[0]
        for i in range(num_conv):
            out_channels = self.base_dim[0] // 2**(num_conv-1-i)
            layers = []
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1))
            if i > 0:
                layers.append(self.norm_dict.get(self.norm)(num_features=out_channels))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            conv_blocks[i] = nn.Sequential(*[l for l in layers if l is not None])
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.flatten = nn.Flatten()
        self.logits = nn.Linear(in_features=np.prod(self.base_dim), out_features=1, bias=False)
        if self.return_logits is False:
            self.pred = nn.Sigmoid()
        
    def forward(self, input:Tensor) -> Tensor:
        x = self.conv_blocks(input)
        x = self.flatten(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x


class WGAN_GP(_GAN):
    """Wasserstein generative adversarial networks with gradient penalty.

    Kwargs: Additional arguments:
    + To `_GAN`: `generator`, `discriminator`, `latent_dim`, `image_dim`.
    + To `Trainer`: `device`, `world_size`, `master_rank`.
    """
    def __init__(self, **kwargs):
        # Critic: alias for discriminator
        if 'critic' in kwargs.keys():
            kwargs['discriminator'] = kwargs['critic']
            kwargs.pop('critic')
            
        super().__init__(**kwargs)
        self.critic = self.discriminator

    def compile(self,
        opt_G:torch.optim.Optimizer,
        opt_D:torch.optim.Optimizer,
        loss_fn_adversarial:bool|Callable[[Any], Tensor]=True,
        loss_fn_gradpen:bool|Callable[[Any], Tensor]=True,
        coeff_ad:float=1,
        coeff_gp:float=10,
        repeat_D:int=5,
        **kwargs,
    ):
        """Compile WGAN-GP.

        Args:
        + `opt_G`: Optimizer for generator.
        + `opt_D`: Optimizer for discriminator/critic.
        + `loss_fn_adversarial`: Adversarial loss function. Pass in a custom    \
            function or toggle with `True`/`False`. Defaults to `True` to use   \
            `WassersteinLoss()`.
        + `loss_fn_gradpen`: Gradient penalty function. Pass in a custom        \
            function or toggle with `True`/`False`. Defaults to `True` to use   \
            `GradientPenalty1LipschitzLoss()`.
        + `coeff_ad`: Multiplier assigned to adversarial loss. Defaults to `1`.
        + `coeff_gp`: Multiplier assigned to gradient penalty. Defaults to `10`.
        + `repeat_D`: Number of discriminator/critic updates per a generator    \
            update iteration. Defaults to `5`.
        
        Kwargs: Additional arguments to `_GAN().compile`: `sync_ddp_metrics`.
        """
        super().compile(**kwargs)
        self.opt_G               = opt_G
        self.opt_D               = opt_D
        self.loss_fn_adversarial = loss_fn_adversarial
        self.loss_fn_gradpen     = loss_fn_gradpen
        self.coeff_ad            = coeff_ad
        self.coeff_gp            = coeff_gp
        self.repeat_D            = repeat_D

        # Config loss and clamper
        self._loss_fn_adversarial = parse_loss(loss_arg=loss_fn_adversarial, default=WassersteinLoss())
        self._loss_fn_gradpen = parse_loss(loss_arg=loss_fn_gradpen, default=GradientPenalty1LipschitzLoss())
        
        self.counter:int = 0
        # Metrics
        self.train_metrics.update({
            'loss_D_ad': Mean(),
            'loss_D_gp': Mean(),
            'loss_D': Mean(),
            'loss_G': Mean(),
        })
        self.val_metrics.update({
            'acc_real': BinaryAccuracy(threshold=0),
            'acc_synth': BinaryAccuracy(threshold=0),
        })

    def train_batch(self, data:Sequence[Tensor]):
        # Unpack data
        x_real, _ = data
        x_real = x_real.to(self.device)
        batch_size:int = x_real.shape[0]

        # Setting models to train mode for more stable training
        self.generator.train()
        self.discriminator.train()

        # Phase 1 - Training the discriminator (critic)
        self.opt_D.zero_grad()
        ## Forward
        z_synth = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device)
        x_synth = self.generator(z_synth)
        pred_real:Tensor = self.discriminator(x_real)
        pred_synth:Tensor = self.discriminator(x_synth)
        loss_D_ad = self._loss_fn_adversarial(input=pred_synth, target=pred_real)
        loss_D_gp = self._loss_fn_gradpen(discriminator=self.discriminator, x_real=x_real, x_synth=x_synth)
        ## Back-propagation
        loss_D:Tensor = (
            + self.coeff_ad*loss_D_ad
            + self.coeff_gp*loss_D_gp
        )
        loss_D.backward()
        self.opt_D.step()

        with torch.inference_mode():
            # Critic's metrics
            self.train_metrics['loss_D_ad'].update(new_entry=loss_D_ad)
            self.train_metrics['loss_D_gp'].update(new_entry=loss_D_gp)
            self.train_metrics['loss_D'].update(new_entry=loss_D)

        # Phase 2 - Training the generator
        # Skip if haven't waited enough `repeat_D` iterations
        self.counter += 1
        if self.counter % self.repeat_D == 0:
            self.counter = 0

            self.opt_G.zero_grad()
            ## Forward
            z_synth = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device)
            x_synth = self.generator(z_synth)
            pred_synth = self.discriminator(x_synth)
            loss_G:Tensor = -self._loss_fn_adversarial(input=pred_synth, target=torch.zeros_like(pred_synth.detach()))
            # Back-propagation
            loss_G.backward()
            self.opt_G.step()
            
            with torch.inference_mode():
                # Generator's metrics
                self.train_metrics['loss_G'].update(new_entry=loss_G)
    
    def test_batch(self, data:Sequence[Tensor]):
        # Unpack data
        x_real, _ = data
        x_real = x_real.to(self.device)
        batch_size:int = x_real.shape[0]

        self.generator.eval()
        self.discriminator.eval()
        with torch.inference_mode():
            # Test 1 - Discriminator's performance on real images
            pred_real = self.discriminator(x_real)
            # Test 2 - Discriminator's performance on synthetic images
            z_synth = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device)
            x_synth = self.generator(z_synth)
            pred_synth = self.discriminator(x_synth)
            # Metrics
            self.val_metrics['acc_real'].update(prediction=pred_real, label=torch.zeros_like(pred_real))
            self.val_metrics['acc_synth'].update(prediction=-pred_synth, label=torch.zeros_like(pred_synth))

    # WARNING: DEPRECATED ######################################################
    def gradient_penalty(self, x_real:Tensor, x_synth:Tensor) -> Tensor:
        '''This method is deprecated. Please use `GradientPenalty1LipschitzLoss`
        and see the comments inside.
        '''
        # Create interpolated images of real and synthetic images
        epsilon = torch.randn(
            size=[x_real.shape[0], *[1]*len(self.image_dim)],
            device=self.device
        )
        epsilon = epsilon.repeat(1, *self.image_dim)

        interpolated = x_real*epsilon + x_synth*(1 - epsilon)
        
        # Compute discriminator scores
        mixed_scores = self.discriminator(interpolated)
        gradient = torch.autograd.grad(
            inputs=interpolated,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradient penalty
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = torch.mean((gradient_norm-1)**2)
        return penalty


if __name__ == '__main__':
    from models.GANs.dcgan import Generator
    from models.GANs.utils import GANGIFMaker, AffineFunction
    from utils.callbacks import CSVLogger, WandBLogger
    from utils.data import get_dataset, get_dataloader

    def expt_mnist(run:int=0):
        LATENT_DIM = 100
        BASE_DIM = [256, 4, 4]
        IMAGE_DIM = [1, 32, 32]
        BATCH_SIZE = 64

        OPT_G, OPT_G_KWARGS = torch.optim.Adam, {'lr': 1e-4, 'betas':[0, 0.9]}
        OPT_D, OPT_D_KWARGS = torch.optim.Adam, {'lr': 1e-4, 'betas':[0, 0.9]}
        COEFF_AD = 1
        COEFF_GP = 10
        REPEAT_D = 5

        NUM_EPOCHS = 50

        dataset = get_dataset(name='MNIST', resize=[32, 32], rescale=[-1, 1])
        dataloader = get_dataloader(dataset=dataset, batch_size=BATCH_SIZE)

        model_G = Generator(latent_dim=LATENT_DIM, base_dim=BASE_DIM, image_dim=IMAGE_DIM)
        model_C = Discriminator(base_dim=BASE_DIM, image_dim=IMAGE_DIM, return_logits=True)

        gan = WGAN_GP(generator=model_G, critic=model_C)
        gan.compile(
            opt_G=OPT_G(params=model_G.parameters(), **OPT_G_KWARGS),
            opt_D=OPT_D(params=model_C.parameters(), **OPT_D_KWARGS),
            loss_fn_adversarial=WassersteinLoss(),
            loss_fn_gradpen=GradientPenalty1LipschitzLoss(),
            coeff_ad=COEFF_AD,
            coeff_gp=COEFF_GP,
            repeat_D=REPEAT_D,
        )
        csv_logger = CSVLogger(
            filename=f'./logs/GANs/{gan.name} - run {run}.csv',
            append=True,
        )
        gif_maker = GANGIFMaker(
            filename=f'./logs/GANs/{gan.name} - run {run}.gif',
            postprocess_fn=AffineFunction(from_range=[-1, 1]),
            save_freq=NUM_EPOCHS//50,
        )
        wandb = WandBLogger(project='wgangp', group='wgangp')
        gan.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['test'],
            callbacks=[csv_logger, gif_maker, wandb],
        )

    for run in range(5):
        expt_mnist(run=run)