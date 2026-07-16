# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Any, Callable, Sequence

from numpy import prod
import torch
from torch import nn, Tensor

from models.GANs._gan import _GAN
from utils.modules import Reshape
from utils.metrics import Mean


class Generator(nn.Module):
    """Fully-connected generator for Generative Adversarial Networks.
            
    Args:
    + `latent_dim`: Dimension of latent space. Defaults to `100`.
    + `image_dim`: Dimension of synthetic images. Defaults to `[1, 28, 28]`.
    """
    def __init__(self,
        latent_dim:int=100,
        image_dim:Sequence[int]=[1, 28, 28],
    ):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_dim = image_dim

        def define_block(in_features:int, out_features:int):
            return nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(num_features=out_features, momentum=0.8),
            )
        
        self.block_0  = define_block(in_features=self.latent_dim, out_features=256)
        self.block_1  = define_block(in_features=256, out_features=512)
        self.block_2  = define_block(in_features=512, out_features=1024)
        self.linear_3 = nn.Linear(in_features=1024, out_features=prod(self.image_dim))
        self.reshape  = Reshape(out_shape=self.image_dim)
        self.tanh     = nn.Tanh()
    
    def forward(self, x:Tensor) -> Tensor:
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.linear_3(x)
        x = self.reshape(x)
        x = self.tanh(x)
        return x
    
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def extra_repr(self) -> str:
        params = {
            'latent_dim':self.latent_dim,
            'image_dim':self.image_dim,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class Discriminator(nn.Module):
    """Fully-connected discriminator for Generative Adversarial Networks.

    Args:
    + `image_dim`: Dimension of input image. Defaults to `[1, 28, 28]`.
    + `return_logits`: Flag to choose between return logits or probability.     \
        Defaults to `True`.
    """
    def __init__(self,
        image_dim:Sequence[int]=[1, 28, 28],
        return_logits:bool=True,
    ):
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

        super(Discriminator, self).__init__()
        self.image_dim = image_dim
        self.return_logits = return_logits

        self.flatten = nn.Flatten()
        def define_block(in_features:int, out_features:int):
            return nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.LeakyReLU(negative_slope=0.2),
            )
        self.block_0 = define_block(in_features=prod(self.image_dim), out_features=512)
        self.block_1 = define_block(in_features=512, out_features=256)
        self.logits = nn.Linear(in_features=256, out_features=1)
        if self.return_logits is False:
            self.pred = nn.Sigmoid()

    def forward(self, x:Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x

    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def extra_repr(self) -> str:
        params = {
            'image_dim':self.image_dim,
            'return_logits':self.return_logits,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class GAN(_GAN):
    """Generative Adversarial Networks.
    
    Kwargs:
    + `generator`: Generator model.
    + `discriminator`: Discriminator model.
    + `latent_dim`: Dimension of latent space. Defaults to `None`, skip to parse\
        from generator.
    + `image_dim`: Dimension of synthetic image. Defaults to `None`, skip to    \
        parse from generator.
    + `device`: The desired device of trainer, needs to be declared explicitly  \
        in case of Distributed Data Parallel training. Defaults to `None`, skip \
        to automatically choose single-process `'cuda'` if available or else    \
        `'cpu'`.
    + `world_size`: The number of processes in case of Distributed Data Parallel\
        training. Defaults to `None`, skips to query automatically with `torch`.
    + `master_rank`: Rank of the master process. Defaults to `0`.
    
    Generative Adversarial Networks - Goodfellow et al.
    DOI: 10.48550/arXiv.1406.2661
    """
    def compile(self,
        opt_G:torch.optim.Optimizer,
        opt_D:torch.optim.Optimizer,
        loss_fn:bool|Callable[[Any], Tensor]=True,
        **kwargs,
    ):
        """Compile GAN.
        
        Args:
        + `opt_G`: Optimizer for generator model.
        + `opt_D`: Optimizer for discriminator model.
        + `loss_fn`: Loss function. Pass in a custom function or toggle with    \
            `True`|`False`. Defaults to `True` to use `nn.BCELoss()`.
            
        Kwargs:
        + `sync_ddp_metrics`: Flag to synchronize metrics during Distributed    \
            Data Parallel training. Can be of type `bool` or dict of `bool`,    \
            accessible with the keys `'train'`|`'val'`. Defaults to `None` for  \
            off for training and on for validation. Warning: Turning this on    \
            will incur extra communication overhead.
        """
        super(GAN, self).compile(**kwargs)
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.loss_fn = loss_fn

        # Config loss
        if self.loss_fn is True:
            self._loss_fn = nn.BCELoss()
        elif self.loss_fn is False:
            self._loss_fn = lambda *args, **kwargs:torch.tensor(0)
        else:
            self._loss_fn = self.loss_fn

        # Metrics
        self.train_metrics.update({
            'loss_D_real': Mean(),
            'loss_D_synth': Mean(),
            'loss_G': Mean(),
        })

    def train_batch(self, data:Sequence[Tensor]):
        # Unpack data
        x_real, _ = data
        x_real = x_real.to(self.device)
        batch_size = x_real.shape[0]
        y_synth = torch.zeros(size=(batch_size, 1), device=self.device)
        y_real = torch.ones(size=(batch_size, 1), device=self.device)

        # Setting models to train mode for more stable training
        self.generator.train()
        self.discriminator.train()

        # Phase 1 - Training discriminator
        self.opt_D.zero_grad()
        ## Forward
        z_synth = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device)
        x_synth = self.generator(z_synth)
        pred_real = self.discriminator(x_real)
        pred_synth = self.discriminator(x_synth)
        loss_D_real = self._loss_fn(pred_real, y_real)
        loss_D_synth = self._loss_fn(pred_synth, y_synth)
        ## Back-propagation
        loss_D_real.backward()
        loss_D_synth.backward()
        self.opt_D.step()

        # Phase 2 - Training generator
        self.opt_G.zero_grad()
        ## Forward
        z_synth = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device)
        x_synth = self.generator(z_synth)
        pred_synth = self.discriminator(x_synth)
        loss_G = self._loss_fn(pred_synth, y_real)
        # Back-propagation
        loss_G.backward()
        self.opt_G.step()

        with torch.inference_mode():
            # Metrics
            self.train_metrics['loss_D_real'].update(new_entry=loss_D_real)
            self.train_metrics['loss_D_synth'].update(new_entry=loss_D_synth)
            self.train_metrics['loss_G'].update(new_entry=loss_G)


if __name__ == '__main__':
    from utils.data import get_dataset, get_dataloader
    from utils.callbacks import CSVLogger
    from models.GANs.utils import MakeGANGIFCallback
    
    def expt_mnist():
        LATENT_DIM = 100
        IMAGE_DIM = [1, 28, 28]
        BATCH_SIZE = 256
        NUM_EPOCHS = 5
        OPT_G, OPT_G_KWARGS = torch.optim.Adam, {'lr': 2e-4, 'betas':(0.5, 0.999)}
        OPT_D, OPT_D_KWARGS = torch.optim.Adam, {'lr': 2e-4, 'betas':(0.5, 0.999)}

        dataset = get_dataset(name='MNIST', resize=IMAGE_DIM[1:], rescale=[-1, 1])
        dataloader = get_dataloader(
            dataset=dataset,
            batch_size={'train': BATCH_SIZE, 'test': 1024},
            shuffle={'train': True, 'test': False},
        )

        model_G = Generator(latent_dim=LATENT_DIM, image_dim=IMAGE_DIM)
        model_D = Discriminator(image_dim=IMAGE_DIM, return_logits=False)
        
        gan = GAN(generator=model_G, discriminator=model_D)
        gan.compile(
            opt_G=OPT_G(params=model_G.parameters(), **OPT_G_KWARGS),
            opt_D=OPT_D(params=model_D.parameters(), **OPT_D_KWARGS),
            loss_fn=nn.BCELoss()
        )

        csv_logger = CSVLogger(
            filename=f'./logs/{gan.name}.csv',
            append=True,
        )
        gif_maker = MakeGANGIFCallback(
            filename=f'./logs/{gan.name}.gif',
            postprocess_fn=lambda x:(x+1)/2,
        )
        gan.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['test'],
            callbacks=[csv_logger, gif_maker],
        )
    
    expt_mnist()