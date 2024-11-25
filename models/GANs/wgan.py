# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Any, Callable, Sequence

import torch
from torch import nn, Tensor

from models.GANs._gan import _GAN
from models.GANs.losses import WassersteinLoss
from utils.metrics import BinaryAccuracy, Mean
from utils.trainers import parse_loss


class WeightClamper():
    """Clamps weight of a module to the range [min, max].

    Args:
    + `min`: Lower bound. Defaults to `-1e-2`.
    + `max`: Upper bound. Defaults to `1e-2`.
    """        
    def __init__(self, min:float=-1e-2, max:float=1e-2):
        self.min = min
        self.max = max

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(min={self.min}, max={self.max})'

    def __call__(self, module:nn.Module):
        if hasattr(module, 'weight'):
            module.weight.data = module.weight.data.clamp(self.min, self.max)


class WGAN(_GAN):
    """Wasserstein generative adversarial networks.

    Kwargs: Additional arguments:
    + To `_GAN`: `generator`, `discriminator`, `latent_dim`, `image_dim`.
    + To `Trainer`: `device`, `world_size`, `master_rank`.
    
    Wasserstein GAN - Arjovsky et al., 2017
    DOI: 10.48550/arXiv.1701.07875 
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
        loss_fn:bool|Callable[[Any], Tensor]=True,
        repeat_D:int=5,
        clamp_weights:bool|Callable[[nn.Module], None]=True,
        **kwargs,
    ):
        """Compile WGAN.
        
        Args:
        + `opt_G`: Optimizer for generator.
        + `opt_D`: Optimizer for discriminator/critic.
        + `loss_fn`: Loss function. Pass in a custom function or toggle with    \
            `True`/`False`. Defaults to `True` to use `WassersteinLoss()`.
        + `repeat_D`: Number of discriminator/critic updates per a generator    \
            update iteration. Defaults to `5`.
        + `clamp_weights`: Weight clamping. Pass in a custom function or toggle \
            with `True`/`False`. Defaults to `True` to clamp to [-0.01, 0.01].
                
        Kwargs: Additional arguments to `Trainer.compile`: `sync_ddp_metrics`.
        """
        super().compile(**kwargs)
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.loss_fn = loss_fn
        self.repeat_D = repeat_D
        self.clamp_weights = clamp_weights

        # Config loss and clamper
        self._loss_fn = parse_loss(loss_arg=loss_fn, default=WassersteinLoss())
        self._clamp_weights = parse_loss(loss_arg=clamp_weights, default=WeightClamper(min=-0.01, max=0.01))
        
        self.counter:int = 0
        # Metrics
        self.train_metrics.update({
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

        self.generator.train()
        self.discriminator.train()
        # Phase 1 - Training the discriminator (critic)
        self.opt_D.zero_grad()
        ## Forward
        z_synth = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device)
        x_synth = self.generator(z_synth)
        pred_real = self.discriminator(x_real)
        pred_synth = self.discriminator(x_synth)
        loss_D:Tensor = self._loss_fn(input=pred_synth, target=pred_real)
        ## Back-propagation
        loss_D.backward()
        self.opt_D.step()

        with torch.inference_mode():
            # Weight clamping
            self._clamp_weights(self.discriminator)
            # Critic's metrics
            self.train_metrics['loss_D'].update(new_entry=loss_D)

        # Phase 2 - Training the generator
        # Skip if haven't waited enough `n_critic` iterations
        self.counter += 1
        if self.counter % self.repeat_D == 0:
            self.counter = 0

            self.opt_G.zero_grad()
            ## Forward
            z_synth = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device)
            x_synth = self.generator(z_synth)
            pred_synth:Tensor = self.discriminator(x_synth)
            loss_G:Tensor = -self._loss_fn(input=pred_synth, target=torch.zeros_like(pred_synth.detach()))
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
    

if __name__ == '__main__':
    from models.GANs.dcgan import Generator, Discriminator
    from models.GANs.utils import GANGIFMaker, AffineFunction
    from utils.callbacks import CSVLogger
    from utils.data import get_dataset, get_dataloader

    def expt_mnist():
        LATENT_DIM = 100
        BASE_DIM = [256, 4, 4]
        IMAGE_DIM = [1, 32, 32]
        BATCH_SIZE = 64
        REPEAT_D = 5
        NUM_EPOCHS = 50
        OPT_G, OPT_G_KWARGS = torch.optim.RMSprop, {'lr': 5e-5}
        OPT_D, OPT_D_KWARGS = torch.optim.RMSprop, {'lr': 5e-5}

        dataset = get_dataset(name='MNIST', resize=[32, 32], rescale=[-1, 1])
        dataloader = get_dataloader(dataset=dataset, batch_size=BATCH_SIZE)

        model_G = Generator(latent_dim=LATENT_DIM, base_dim=BASE_DIM, image_dim=IMAGE_DIM)
        model_D = Discriminator(base_dim=BASE_DIM, image_dim=IMAGE_DIM, return_logits=True)
        gan = WGAN(generator=model_G, discriminator=model_D)
        gan.compile(
            opt_G=OPT_G(params=model_G.parameters(), **OPT_G_KWARGS),
            opt_D=OPT_D(params=model_D.parameters(), **OPT_D_KWARGS),
            loss_fn=WassersteinLoss(),
            repeat_D=REPEAT_D,
            clamp_weights=True,
        )
        
        csv_logger = CSVLogger(
            filename=f'./logs/GANs/{gan.name}.csv',
            append=True,
        )
        gif_maker = GANGIFMaker(
            filename=f'./logs/GANs/{gan.name}.gif',
            postprocess_fn=AffineFunction(from_range=[-1, 1]),
        )
        gan.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['test'],
            callbacks=[csv_logger, gif_maker],
        )

    expt_mnist()