# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Sequence

import numpy as np
import torch
from torch import nn, Tensor

from models.GANs.gan import GAN
from utils.modules import Reshape


class Sum(nn.Module):
    """Sum inputs by specified dimension.

    Args:
    + `dim`: Dimensions to sum.
    """
    def __init__(self, dim:int|Sequence[int]):
        super(Sum, self).__init__()
        if isinstance(dim, int):
            self.dim = [dim]
        else:
            self.dim = dim
    
    def forward(self, input:Tensor) -> Tensor:
        return input.sum(dim=self.dim)

    def extra_repr(self) -> str:
        params = {'dim':self.dim}
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class Generator(nn.Module):
    """Deep convolutional generator for DCGAN.
    
    Args:
    + `latent_dim`: Dimension of latent space. Defaults to `100`.
    + `image_dim`: Dimension of synthetic images. Defaults to `[1, 28, 28]`.
    + `base_dim`: Dimension of the shallowest feature maps. After each          \
        convolutional layer, each dimension is doubled the and number of filters\
        is halved until `image_dim` is reached. Defaults to `[256, 7, 7]`.

    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    def __init__(self,
        latent_dim:int=100,
        image_dim:Sequence[int]=[1, 28, 28],
        base_dim:Sequence[int]=[256, 7, 7],
    ):
        # Parse architecture from input dimension
        dim_ratio = [image_dim[axis]/base_dim[axis] for axis in torch.arange(start=1, end=len(image_dim))]
        for axis in range(len(dim_ratio)):
            num_conv = np.log(dim_ratio[axis])/np.log(2.)
            assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
            assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
        num_conv = int(num_conv)

        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.base_dim = base_dim

        self.linear_0 = nn.Linear(in_features=self.latent_dim, out_features=np.prod(self.base_dim), bias=False)
        self.reshape = Reshape(out_shape=self.base_dim)
        self.bnorm_0 = nn.BatchNorm2d(num_features=self.base_dim[0])
        self.relu_0  = nn.ReLU()

        convt_blocks = [None for i in range(num_conv)]
        for i in range(num_conv):
            in_channels = self.base_dim[0] // 2**i
            out_channels = self.base_dim[0] // 2**(i+1)
            if i < num_conv - 1:
                block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU(),
                )
            elif i == num_conv - 1:
                # Last Conv2DTranspose: not use BatchNorm, replace relu with tanh
                block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=self.image_dim[0], kernel_size=4, stride=2, padding=1, bias=False),
                    nn.Tanh()
                )
            convt_blocks[i] = block
        self.convt_blocks = nn.Sequential(*convt_blocks)

    def forward(self, input:Tensor) -> Tensor:
        x = self.linear_0(input)
        x = self.reshape(x)
        x = self.bnorm_0(x)
        x = self.relu_0(x)
        x = self.convt_blocks(x)
        return x
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def extra_repr(self) -> str:
        params = {
            'latent_dim':self.latent_dim,
            'image_dim':self.image_dim,
            'base_dim':self.base_dim,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class Discriminator(nn.Module):
    """Deep convolutional discriminator for DCGAN. Ideally should have a
    symmetric architecture with the generator's.

    Args:
    + `image_dim`: Dimension of image. Defaults to `[1, 28, 28]`.
    + `base_dim`: Dimension of the shallowest feature maps, ideally equal to the\
        generator's. In contrast, after each convolutional layer, each dimension\
        from `image_dim` is halved and the number of filters is doubled until   \
        `base_dim` is reached. Defaults to `[256, 7, 7]`.
    + `return_logits`: Flag to choose between return logits or probability.     \
        Defaults to `True`.
    
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    def __init__(self,
        image_dim:Sequence[int]=[1, 28, 28],
        base_dim:Sequence[int]=[256, 7, 7],
        return_logits:bool=True,
    ):
        # Parse architecture from input dimension
        dim_ratio = [image_dim[axis]/base_dim[axis] for axis in torch.arange(start=1, end=len(image_dim))]
        for axis in range(len(dim_ratio)):
            num_conv = np.log(dim_ratio[axis])/np.log(2.)
            assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
            assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
        num_conv = int(num_conv)
        
        super(Discriminator, self).__init__()
        self.image_dim = image_dim
        self.base_dim = base_dim
        self.return_logits = return_logits

        self.conv_blocks = [None for i in range(num_conv)]
        for i in range(num_conv):
            in_channels = self.base_dim[0] // 2**(num_conv-i)
            out_channels = self.base_dim[0] // 2**(num_conv-1-i)
            if i == 0:
                # First Conv2D: not use BatchNorm 
                self.conv_blocks[i] = nn.Sequential(
                    nn.Conv2d(in_channels=self.image_dim[0], out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            elif i > 0:
                self.conv_blocks[i] = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.LeakyReLU(negative_slope=0.2)
                )
        self.conv_blocks = nn.Sequential(*self.conv_blocks)

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

    @property
    def name(self) -> str:
        return self.__class__.__name__
 
    def extra_repr(self) -> str:
        params = {
            'image_dim':self.image_dim,
            'base_dim':self.base_dim,
            'return_logits':self.return_logits,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class Generator2(nn.Module):
    """Deep convolutional generator for DCGAN (alternative).
    
    Args:
    + `latent_dim`: Dimension of latent space. Defaults to `100`.
    + `image_dim`: Dimension of synthetic images. Defaults to `[1, 28, 28]`.
    + `base_dim`: Dimension of the shallowest feature maps. After each          \
        convolutional layer, each dimension is doubled the and number of filters\
        is halved until `image_dim` is reached. Defaults to `[256, 7, 7]`.

    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    def __init__(self,
        latent_dim:int=128,
        image_dim:Sequence[int]=[3, 32, 32],
        base_dim:Sequence[int]=[512, 4, 4],
    ):
        # Parse architecture from input dimension
        dim_ratio = [image_dim[axis]/base_dim[axis] for axis in torch.arange(start=1, end=len(image_dim))]
        for axis in range(len(dim_ratio)):
            num_conv = np.log(dim_ratio[axis])/np.log(2.) + 1
            assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
            assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
        num_conv = int(num_conv)

        super(Generator2, self).__init__()
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.base_dim = base_dim

        self.linear  = nn.Linear(in_features=self.latent_dim, out_features=np.prod(self.base_dim), bias=True)
        self.reshape = Reshape(out_shape=self.base_dim)
        
        convt_blocks = []
        for i in range(num_conv):
            in_channels  = self.base_dim[0] // 2**i
            out_channels = self.base_dim[0] // 2**(i+1)
            if i < num_conv - 1:
                block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=out_channels, eps=1e-4, momentum=0.1, affine=True),
                    nn.ReLU(),
                )
            elif i == num_conv - 1:
                # Last Conv2DTranspose: not use BatchNorm, replace relu with tanh
                block = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=self.image_dim[0], kernel_size=3, stride=1, padding=1, bias=True),
                    nn.Tanh()
                )
            convt_blocks.append(block)
        self.convt_blocks = nn.Sequential(*convt_blocks)

    def forward(self, x:Tensor) -> Tensor:
        x = self.linear(x)
        x = self.reshape(x)
        x = self.convt_blocks(x)
        return x

    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def extra_repr(self) -> str:
        params = {
            'latent_dim':self.latent_dim,
            'image_dim':self.image_dim,
            'base_dim':self.base_dim,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class Discriminator2(nn.Module):
    """Deep convolutional discriminator for DCGAN (alternative). Ideally should 
    have a symmetric architecture with the generator's.

    Args:
    + `image_dim`: Dimension of image. Defaults to `[1, 28, 28]`.
    + `base_dim`: Dimension of the shallowest feature maps, ideally equal to the\
        generator's. In contrast, after each convolutional layer, each dimension\
        from `image_dim` is halved and the number of filters is doubled until   \
        `base_dim` is reached. Defaults to `[256, 7, 7]`.
    + `return_logits`: Flag to choose between return logits or probability.     \
        Defaults to `True`.
    
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    def __init__(self,
        image_dim:Sequence[int]=[3, 32, 32],
        base_dim:Sequence[int]=[512, 4, 4],
        return_logits:bool=True,
    ):
        # Parse architecture from input dimension
        dim_ratio = [image_dim[axis]/base_dim[axis] for axis in torch.arange(start=1, end=len(image_dim))]
        for axis in range(len(dim_ratio)):
            num_conv = np.log(dim_ratio[axis])/np.log(2.)
            assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
            assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
        num_conv = int(num_conv)

        super(Discriminator2, self).__init__()
        self.image_dim = image_dim
        self.base_dim = base_dim
        self.return_logits = return_logits

        self.conv_blocks = []
        for i in range(num_conv):
            in_channels = self.image_dim[0] if (i == 0) else self.base_dim[0] // 2**(num_conv+1-i)
            out_channels = self.base_dim[0] // 2**(num_conv-i)
            # First Conv2D: not use BatchNorm 
            block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(num_features=out_channels, eps=1e-4, momentum=0.1, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(num_features=out_channels, eps=1e-4, momentum=0.1, affine=True),
                nn.ReLU(),
            )
            self.conv_blocks.append(block)
        self.conv_blocks = nn.Sequential(*self.conv_blocks)

        self.conv   = nn.Conv2d(in_channels=out_channels, out_channels=self.base_dim[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.bnorm  = nn.BatchNorm2d(num_features=self.base_dim[0], eps=1e-4, momentum=0.1, affine=True)
        self.relu   = nn.ReLU()
        self.sum    = Sum(dim=[2, 3])
        self.logits = nn.Linear(in_features=self.base_dim[0], out_features=1, bias=True)
        if self.return_logits is False:
            self.pred = nn.Sigmoid()

    def forward(self, x:Tensor) -> Tensor:
        x = self.conv_blocks(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.sum(x)
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
            'base_dim':self.base_dim,
            'return_logits':self.return_logits,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class DCGAN(GAN):
    """Deep convolutional Generative Adversarial Networks.
    
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
    
    Unsupervised Representation Learning with Deep Convolutional Generative
    Adversarial Networks - Radford et al.
    DOI: 10.48550/arXiv.1511.06434
    """
    pass


if __name__ == '__main__':
    from torchinfo import summary
    from utils.data import get_dataloader
    from utils.callbacks import CSVLogger
    from models.GANs.utils import MakeSyntheticGIFCallback

    def expt_summary():
        model = Discriminator2()
        summary(model, [100, 3, 32, 32])

    expt_summary()

    def expt_mnist():
        LATENT_DIM = 100
        IMAGE_DIM = [1, 28, 28]
        BASE_DIM = [256, 7, 7]
        BATCH_SIZE = 128
        NUM_EPOCHS = 50
        OPT_G, OPT_G_KWARGS = torch.optim.Adam, {'lr': 2e-4, 'betas':(0.5, 0.999)}
        OPT_D, OPT_D_KWARGS = torch.optim.Adam, {'lr': 2e-4, 'betas':(0.5, 0.999)}

        dataloader = get_dataloader(
            dataset='MNIST',
            rescale=[-1, 1],
            batch_size_train=BATCH_SIZE,
        )

        model_G = Generator(
            latent_dim=LATENT_DIM,
            image_dim=IMAGE_DIM,
            base_dim=BASE_DIM,
        )
        model_D = Discriminator(
            image_dim=IMAGE_DIM,
            base_dim=BASE_DIM,
            return_logits=False,
        )

        gan = DCGAN(generator=model_G, discriminator=model_D)
        gan.compile(
            optimizer_G=OPT_G(params=model_G.parameters(), **OPT_G_KWARGS),
            optimizer_D=OPT_D(params=model_D.parameters(), **OPT_D_KWARGS),
            loss_fn=nn.BCELoss(),
        )

        csv_logger = CSVLogger(
            filename=f'./logs/{gan.name}.csv',
            append=True,
        )
        gif_maker = MakeSyntheticGIFCallback(
            filename=f'./logs/{gan.name}.gif',
            postprocess_fn=lambda x:(x+1)/2,
        )
        gan.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['test'],
            callbacks=[csv_logger, gif_maker],
        )
    
    def expt_cifar():
        LATENT_DIM = 128
        IMAGE_DIM = [3, 32, 32]
        BASE_DIM = [512, 4, 4]
        BATCH_SIZE = 64
        NUM_EPOCHS = 100
        G_CLASS, G_KWARGS = Generator2, {}
        D_CLASS, D_KWARGS = Discriminator2, {}
        OPT_G, OPT_G_KWARGS = torch.optim.Adam, {'lr': 2e-4, 'betas':(0.5, 0.999)}
        OPT_D, OPT_D_KWARGS = torch.optim.Adam, {'lr': 2e-4, 'betas':(0.5, 0.999)}

        dataloader = get_dataloader(
            dataset='CIFAR10',
            rescale=[-1, 1],
            batch_size_train=BATCH_SIZE,
        )

        model_G = G_CLASS(**G_KWARGS, latent_dim=LATENT_DIM, image_dim=IMAGE_DIM, base_dim=BASE_DIM)
        model_D = D_CLASS(**D_KWARGS, image_dim=IMAGE_DIM, base_dim=BASE_DIM, return_logits=False)
        gan = DCGAN(generator=model_G, discriminator=model_D)
        gan.compile(
            opt_G=OPT_G(params=model_G.parameters(), **OPT_G_KWARGS),
            opt_D=OPT_D(params=model_D.parameters(), **OPT_D_KWARGS),
            loss_fn=nn.BCELoss(),
        )

        csv_logger = CSVLogger(
            filename=f'./logs/{gan.name}.csv',
            append=True,
        )
        gif_maker = MakeSyntheticGIFCallback(
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