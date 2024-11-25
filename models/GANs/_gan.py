# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Sequence, Optional

import torch
from torch import nn, Tensor

from utils.trainers import Trainer
from utils.metrics import BinaryAccuracy


class _GAN(Trainer):
    """Base class for generative adversarial networks.

    Args:
    + `generator`: Generator model.
    + `discriminator`: Discriminator model.
    + `latent_dim`: Dimension of latent space. Defaults to `None`, skip to parse\
        from generator.
    + `image_dim`: Dimension of synthetic image. Defaults to `None`, skip to    \
        parse from generator.
    
    Kwargs:
    + `device`: The desired device of trainer, needs to be declared explicitly  \
        in case of Distributed Data Parallel training. Defaults to `None`, skip \
        to automatically choose single-process `'cuda'` if available or else    \
        `'cpu'`.
    + `world_size`: The number of processes in case of Distributed Data Parallel\
        training. Defaults to `None`, skips to query automatically with `torch`.
    + `master_rank`: Rank of the master process. Defaults to `0`.
    """
    def __init__(self,
        generator:nn.Module,
        discriminator:nn.Module,
        latent_dim:Optional[int]=None,
        image_dim:Optional[Sequence[int]]=None,
        **kwargs,
    ):
        super(_GAN, self).__init__(**kwargs)
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.latent_dim = latent_dim
        self.image_dim = image_dim

        if self.latent_dim is None:
            self.latent_dim:int = self.generator.latent_dim

        if self.image_dim is None:
            self.image_dim:int = self.generator.image_dim

    def compile(self, **kwargs):
        """Compile GAN.
        
        Kwargs:
        + `sync_ddp_metrics`: Flag to synchronize metrics during Distributed    \
            Data Parallel training. Can be of type `bool` or dict of `bool`,    \
            accessible with the keys `'train'`|`'val'`. Defaults to `None` for  \
            off for training and on for validation. Warning: Turning this on    \
            will incur extra communication overhead.
        """
        super(_GAN, self).compile(**kwargs)
        
        self.val_metrics.update({
            'acc_real': BinaryAccuracy(),
            'acc_synth': BinaryAccuracy(),
        })

    def train_batch(self, data:Sequence[Tensor]):
        raise NotImplementedError('Subclasses should be implemented first.')

    def test_batch(self, data:Sequence[Tensor]):
        # Unpack data
        x_real, _ = data
        x_real = x_real.to(self.device)
        batch_size:int = x_real.shape[0]
        y_synth = torch.zeros(size=(batch_size, 1), device=self.device)
        y_real = torch.ones(size=(batch_size, 1), device=self.device)

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
            self.val_metrics['acc_real'].update(label=y_real, prediction=pred_real)
            self.val_metrics['acc_synth'].update(label=y_synth, prediction=pred_synth)