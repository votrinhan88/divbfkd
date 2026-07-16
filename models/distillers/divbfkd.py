# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from copy import deepcopy
from typing import Any, Callable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn, Tensor
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torchvision import transforms
import tqdm.auto as tqdm
from umap import UMAP
import yaml

from models.classifiers import ClassifierTrainer, AlexNet, LeNet5, VGG
from models.classifiers.resnetcifar import ResNet_CIFAR
from models.distillers.distiller import Distiller
from models.distillers.standardkd import StandardKD
from models.distillers.utils import IntermediateFeatureExtractor
from models.GANs.dcgan import Discriminator
from models.GANs.utils import GANGIFMaker
from models.GANs.wgan import WassersteinLoss, WeightClamper
from models.GANs.wgan_gp import WGAN_GP
from utils.callbacks import Callback, ProgressBar, CSVLogger, ModelCheckpoint, LearningRateSchedulerOnEpoch, SchedulerOnEpoch

from utils.config.pprint import pprint
from utils.data import show_aug, NdArrayPool, TensorPool, get_dataloader, get_dataset, get_transform, make_fewshot
from utils.data.metadata import METADATA
from utils.ddp import all_gather_nd, cleanup, setup_master, setup_process
from utils.metrics import CategoricalAccuracy, Counter, DDPMetric, Mean, MultiCounter, Scalar
from utils.modules import Reshape


class Generator(nn.Module):
    def __init__(self,
        latent_dim:int=256,
        image_dim:Sequence[int]=[3, 32, 32],
        base_dim:Sequence[int]=[256, 8, 8],
        NormLayer:nn.Module=nn.BatchNorm2d,
    ):
        # Parse architecture from input dimension
        dim_ratio = [image_dim[axis]/base_dim[axis] for axis in torch.arange(start=1, end=len(image_dim))]
        for axis in range(len(dim_ratio)):
            num_conv = np.log(dim_ratio[axis])/np.log(2.)
            assert num_conv == round(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
            assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
        num_conv = round(num_conv)

        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.base_dim = base_dim
        self.NormLayer = NormLayer

        self.linear_1 = nn.Linear(in_features=self.latent_dim, out_features=np.prod(self.base_dim))
        self.reshape = Reshape(out_shape=self.base_dim)
        self.conv_in = self.NormLayer(num_features=self.base_dim[0])
        
        conv_blocks = [None]*num_conv
        for i in range(num_conv):
            if i == 0:
                in_channels = self.base_dim[0] // 2**i
                out_channels = in_channels
            elif i > 0:
                in_channels = self.base_dim[0] // 2**(i-1)
                out_channels = self.base_dim[0] // 2**i

            conv_blocks[i] = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                self.NormLayer(num_features=out_channels),
                nn.LeakyReLU(negative_slope=0.2),
            )
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.conv_out = nn.Conv2d(in_channels=out_channels, out_channels=self.image_dim[0], kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input:Tensor) -> Tensor:
        x = self.linear_1(input)
        x = self.reshape(x)
        x = self.conv_in(x)
        x = self.conv_blocks(x)
        x = self.conv_out(x)
        x = self.sigmoid(x)
        return x
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def extra_repr(self) -> str:
        params = {
            'latent_dim':self.latent_dim,
            'image_dim':self.image_dim,
            'base_dim':self.base_dim,
            'NormLayer':self.NormLayer,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class DivBFKD(Distiller):
    def __init__(self,
        generator:nn.Module,
        discriminator:nn.Module,
        num_classes:Optional[int]=None,
        latent_dim:Optional[int]=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = generator.to(device=self.device)
        self.discriminator = discriminator.to(device=self.device)
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        if self.latent_dim is None:
            self.latent_dim:int = self.generator.latent_dim
        
        if self.num_classes is None:
            self.num_classes:int = self.student.module.num_classes if self.ddp else self.student.num_classes
        
        if self.ddp:
            self.generator = DDP(module=generator, device_ids=[self.rank])
            self.discriminator = DDP(module=discriminator, device_ids=[self.rank], broadcast_buffers=False)

    def compile(self,
        opt_G:torch.optim.Optimizer,
        opt_D:torch.optim.Optimizer,
        opt_S:torch.optim.Optimizer,
        filter:dict={},
        gan_pool_config:dict={},
        loss_fn_adversarial:bool|Callable[[Any], Tensor]=True,
        loss_fn_regularize:bool|Callable[[Any], Tensor]=True,
        loss_fn_distill:bool|Callable[[Any], Tensor]=True,
        coeff_D_ad:float=1,
        coeff_D_rg:float=0.1,
        coeff_D_gp:float=10,
        repeat_D:int=5,
        clamp_weights:bool|Callable[[nn.Module], None]=False,
        aug_S:Optional[Callable[[Any], Tensor]]=None,
        temperature:float=1,
        distill_pool_config:Optional[dict]=None,
        batch_size:Optional[int]=250,
        **kwargs,
    ):
        """
        Kwargs:
        + `loss_fn_test`: Test loss function to test student's prediction with  \
            ground truth label, of type `bool` or custom. `True`: use the cross-\
            entropy loss, `False`: skip measuring test loss, or pass in a custom\
            loss function. Defaults to `True`.
        """        
        super().compile(**kwargs)
    
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.opt_S = opt_S
        self.filter = filter
        self.gan_pool_config = gan_pool_config
        self.loss_fn_adversarial = loss_fn_adversarial
        self.loss_fn_regularize = loss_fn_regularize
        self.loss_fn_distill = loss_fn_distill
        self.coeff_D_ad = coeff_D_ad
        self.coeff_D_rg = coeff_D_rg
        self.coeff_D_gp = coeff_D_gp
        self.repeat_D = repeat_D
        self.clamp_weights = clamp_weights
        self.temperature = temperature
        self.distill_pool_config = distill_pool_config
        self.aug_S = aug_S
        self.batch_size = batch_size

        # Config advesarial loss
        if self.loss_fn_adversarial == True:
            self._loss_fn_adversarial = WassersteinLoss(reduction='mean')
        elif self.loss_fn_adversarial == False:
            self._loss_fn_adversarial = lambda *args, **kwargs:torch.tensor(0)
        else:
            self._loss_fn_adversarial = self.loss_fn_adversarial
        # Config regularize loss
        if self.loss_fn_regularize == True:
            self._loss_fn_regularize = nn.CrossEntropyLoss()
        elif self.loss_fn_regularize == False:
            self._loss_fn_regularize = lambda *args, **kwargs:torch.tensor(0)
        else:
            self._loss_fn_regularize = self.loss_fn_regularize
        # Config distillation loss
        if self.loss_fn_distill == True:
            self._loss_fn_distill = nn.CrossEntropyLoss()
        elif self.loss_fn_distill == False:
            self._loss_fn_distill = lambda *args, **kwargs:torch.tensor(0)
        else:
            self._loss_fn_distill = self.loss_fn_distill

        # Config weight clamper        
        if self.clamp_weights == True:
            self._clamp_weights = WeightClamper(min=-0.01, max=0.01)
        elif self.clamp_weights == False:
            self._clamp_weights = lambda *args, **kwargs:None
        else:
            self._clamp_weights = self.clamp_weights

        # Config augmentation
        if self.aug_S is None:
            self.aug_S = nn.Identity()

        # Counter for WGAN
        self.counter:int = 0

        # Metrics
        self.train_metrics.update({
            'loss_D': Mean(),
            'loss_D_ad': Mean(),
            'loss_D_gp': Mean(),
            'loss_D_rg': Mean(),
            'loss_G': Mean(),
            'loss_G_ad': Mean(),
            'loss_S': Mean(),
            'loss_dt': Mean(),
            'acc': CategoricalAccuracy(onehot_label=True),
            'expected_r':Mean(),
            'expected_s':Mean(),
            'confidence_r':Mean(),
            'confidence_s':Mean(),
            'pool_size': Scalar(persist=True),
            'stage': Scalar(persist=True),
            'query': Counter(cumulate=True),
            'count_S': MultiCounter(num_classes=self.num_classes, reduction='argmax', cumulate=False),
            'count_T': MultiCounter(num_classes=self.num_classes, reduction='argmax', cumulate=False),
        })
        if self.gan_pool_config.get('reset') == True:
            self.train_metrics.update({'pool_fake':Scalar(persist=False)})
        # Wrap metrics in DDP
        if self.ddp:
            for metric in [
                'acc',
                'loss_D', 'loss_D_ad', 'loss_D_gp', 'loss_D_rg',
                'loss_G', 'loss_G_ad', 'loss_S', 'loss_dt',
                'expected_r', 'expected_s', 'confidence_r', 'confidence_s',
                'pool_size', 'pool_fake', 'stage', 'query', 'count_T', 'count_S',
            ]:
                self.train_metrics.update({metric:DDPMetric(
                    metric=self.train_metrics.get(metric), 
                    device=self.device, 
                    world_size=self.world_size,
                )})

        self.fewshot_pool:Optional[NdArrayPool] = None
        self.gan_pool    :Optional[TensorPool] = None
        self.distill_pool:Optional[ConcatDataset] = None

    def on_epoch_begin(self, epoch:int, logs=None):
        super().on_epoch_begin(epoch, logs)
    
        if (self.train_metrics['stage'].value == 1) & (self.gan_pool_config.get('filter') == True):
            # Reset GAN pool to fewshot data ###################################
            if self.gan_pool_config.get('reset') == True:
                fewshot_cache = next(iter(self.fewshot_cache_loader))
                self.gan_pool.delete()
                self.gan_pool.add(new_data=[fewshot_cache[0], fewshot_cache[1]])

            # Candidates: Shuffle and cap size to 100 to save memory ##########
            for k in range(self.num_classes):
                # Gather candidates from all process
                if self.ddp:
                    candidates_k = self.candidates[k].fields[0].to(device=self.device)
                    candidates_k_list = all_gather_nd(tensor=candidates_k, world_size=self.world_size)
                    self.candidates[k].delete()
                    self.candidates[k].add([torch.cat(tensors=candidates_k_list, dim=0)])
                # Shuffle and cap size to 100
                if len(self.candidates[k]) > 100:
                    if (not self.ddp) or (self.ddp & self.rank == self.master_rank):
                        slice_indices = torch.randperm(len(self.candidates[k]))[0:100]
                    else:
                        slice_indices = torch.empty([100])
                    if self.ddp:
                        dist.broadcast(tensor=slice_indices, src=self.master_rank)

                    self.candidates[k].slice(indices=slice_indices)

            # Add synthetic images to GAN pool #################################
            num_fakes = 0
            if self.gan_pool_config.get('balanced_classes') == True:
                # Balanced classes (batches of class [0..k])
                add_amount = np.min([len(c) for c in self.candidates])
                if add_amount > 0:
                    x_synth = torch.cat(
                        tensors=[
                            self.candidates[k].fields[0][0:add_amount]
                                for k in range(self.num_classes)
                        ],
                        dim=0,
                    )
                    y_synth = torch.cat(
                        tensors=[
                            k*torch.ones(size=[add_amount], dtype=torch.int64)
                                for k in range(self.num_classes)
                        ],
                        dim=0,
                    )
                    # TODO: Check shape and correctness of x_synth, y_synth
                    for k in range(self.num_classes):
                        self.candidates[k].delete(torch.arange(add_amount))
                    balanced_indices = (
                        torch.arange(self.num_classes*add_amount)
                            .reshape(self.num_classes, add_amount).T.ravel()
                    )

                    self.gan_pool.add([
                        x_synth[balanced_indices],
                        y_synth[balanced_indices],
                    ])
                num_fakes = add_amount*self.num_classes
            else:
                # Any examples that pass the filter
                for k in range(self.num_classes):
                    add_amount = len(self.candidates[k])
                    # TODO: Check shape of candidates
                    self.gan_pool.add([
                        self.candidates[k].fields[0],
                        k*torch.ones(size=[add_amount], dtype=self.gan_pool.dtypes[1]),
                    ])
                    num_fakes = num_fakes + add_amount
                    self.candidates[k].delete()
            
            if self.ddp:
                self.train_metrics['pool_fake'].metric.update(num_fakes/(num_fakes + len(self.gan_pool)))
            else:
                self.train_metrics['pool_fake'].update(num_fakes/(num_fakes + len(self.gan_pool)))

            # GAN pool: shuffle (optionally by class), and cap by `cap_size` ###
            cap_size = len(self.fewshot_pool) + self.gan_pool_config.get('budget')
            if (not self.ddp) or (self.ddp & self.rank == self.master_rank):
                if self.gan_pool_config.get('balanced_classes') == True:
                    # Shuffle indices class-wise
                    idx_shuffle = torch.arange(len(self.gan_pool), device=self.device)
                    for k in range(self.num_classes):
                        idx_k = (self.gan_pool.fields[1] == k).nonzero().squeeze(dim=1)
                        idx_k_shuffled = idx_k[torch.randperm(idx_k.shape[0])]
                        idx_shuffle[idx_k] = idx_shuffle[idx_k_shuffled]
                else:
                    # Shuffle indices randomly
                    idx_shuffle = torch.randperm(len(self.gan_pool), device=self.device)
                # Cap size
                idx_shuffle = idx_shuffle[0:cap_size]
            else:
                idx_shuffle = torch.empty(size=[cap_size], device=self.device, dtype=torch.long)

            if self.ddp:
                dist.broadcast(tensor=idx_shuffle, src=self.master_rank)
            
            self.gan_pool.slice(idx_shuffle.to(device=self.gan_pool.device))
            self.train_metrics['pool_size'].update(len(self.gan_pool))

    def train_batch_1(self, data:Sequence[Tensor]):
        # Unpack data
        x_real, y_real = data
        x_real = x_real.to(self.device)
        y_real = y_real.to(self.device)
        batch_size:int = x_real.shape[0]

        # Setting models to train mode for more stable training
        self.generator.train()
        self.discriminator.train()

        # Phase 1 - Training the discriminator
        self.opt_D.zero_grad()
        ## Forward
        z_synth = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device)
        x_synth = self.generator(z_synth)

        logits_D_real:Tensor = self.discriminator(x_real)
        logits_D_synth:Tensor = self.discriminator(x_synth)
        prob_T_synth = self.blackbox(self.teacher(x_synth))

        loss_D_ad = self._loss_fn_adversarial(input=logits_D_synth, target=logits_D_real)
        loss_D_rg = self._loss_fn_regularize(input=logits_D_synth.sigmoid(), target=prob_T_synth.max(dim=1, keepdim=True)[0])
        loss_D_gp = self.gradient_penalty(x_real=x_real, x_synth=x_synth)
        loss_D = (
            + self.coeff_D_ad*loss_D_ad
            + self.coeff_D_gp*loss_D_gp
            + self.coeff_D_rg*loss_D_rg
        )
        ## Backward
        loss_D.backward()
        self.opt_D.step()

        with torch.inference_mode():
            # Collect confident samples
            if self.gan_pool_config.get('filter') == True:
                confidence, pred_T_synth = prob_T_synth.max(dim=1)
                for k in range(self.num_classes):
                    idx_confident = ((pred_T_synth == k) & (confidence > self.threshold[k])).nonzero().squeeze(dim=1)
                    self.candidates[k].add([x_synth[idx_confident]])
            # Weight clamping
            self._clamp_weights(self.discriminator)
            # Metrics
            self.train_metrics['count_T'].update(prob_T_synth)
            self.train_metrics['query'].update(prob_T_synth.shape[0])
            self.train_metrics['expected_r'].update(logits_D_real.sign().mean(dim=0))
            self.train_metrics['expected_s'].update(-logits_D_synth.sign().mean(dim=0))
            self.train_metrics['confidence_s'].update(prob_T_synth.log().max(dim=1)[0].mean(dim=0))
            
            prob_T_real = self.blackbox(self.teacher(x_real))
            self.train_metrics['confidence_r'].update(prob_T_real.log().max(dim=1)[0].mean(dim=0))

            if self.loss_fn_adversarial is not False:
                self.train_metrics['loss_D_ad'].update(new_entry=loss_D_ad)
            if self.loss_fn_regularize is not False:
                self.train_metrics['loss_D_rg'].update(new_entry=loss_D_rg)
            self.train_metrics['loss_D_gp'].update(new_entry=loss_D_gp)
            self.train_metrics['loss_D'].update(new_entry=loss_D)

        # Phase 2 - Training the generator
        # Skip if haven't waited enough `repeat_D` iterations
        self.counter = self.counter + 1
        if self.counter % self.repeat_D == 0:
            self.counter = 0

            self.opt_G.zero_grad()
            ## Forward
            z_synth = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device)
            x_synth = self.generator(z_synth)

            logits_D_synth = self.discriminator(x_synth)
            loss_G_ad = -self._loss_fn_adversarial(input=logits_D_synth, target=torch.zeros_like(logits_D_synth))
            loss_G = loss_G_ad
            # Back-propagation
            loss_G.backward()
            self.opt_G.step()
            
            with torch.inference_mode():
                # Generator's metrics
                if self._loss_fn_adversarial is not False:
                    self.train_metrics['loss_G_ad'].update(new_entry=loss_G_ad)
                    self.train_metrics['loss_G'].update(new_entry=loss_G)

    def train_batch_2(self, data:Sequence[Tensor]):
        # Unpack data
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        self.teacher.eval()
        self.student.train()
        self.opt_S.zero_grad()
        # Forward
        x_aug = self.aug_S(x)
        logits_S = self.student(x_aug)

        # y_soft = (y.log()/self.temperature).softmax(dim=1) # == y if self.temperature == 1
        y_soft = y
        # Distillation loss: Not multiplying with T^2 gives slightly better results
        loss_distil = self._loss_fn_distill(input=logits_S/self.temperature, target=y_soft)# * self.temperature**2
        loss_S = loss_distil
        
        # Backward
        if self.loss_fn_distill is not False:
            loss_S.backward()
            self.opt_S.step()
        
        # Metrics
        with torch.inference_mode():
            if self.loss_fn_distill is not False:
                self.train_metrics['count_S'].update(logits_S)
                self.train_metrics['loss_dt'].update(new_entry=loss_distil)
                self.train_metrics['loss_S'].update(new_entry=loss_S)
            self.train_metrics['acc'].update(prediction=logits_S, label=y)

    def training_loop(self,
        stage:Optional[int],
        fewshot_pool:Optional[NdArrayPool]=None,
        synth_data:Optional[dict[str, Tensor]|DataLoader]=None,
        **kwargs,
    ):
        """Training loop.

        Args:
        + `stage`: Stage of framework: `1`|`2`.
        + `fewshot_data`: The fewshot dataset, with samples and targets accessible
        via the corresponding attributes.

        Kwargs:
        + `num_epochs`: Number of epochs.
        + `valloader`: Dataloader for validation set. Defaults to `None`.
        + `callbacks`: List of callbacks. Defaults to `None`.
        + `start_epoch`: Index of first epoch. Defaults to `0`.
        + `val_freq`: Number of training epoch to perform a validation step.   \
            Defaults to `1`.
        """
        if 'trainloader' in kwargs.keys():
            raise NotImplementedError(
                'Argument `trainloader` will not be used. Did you mean `fewshot_data`?'
            )

        # FEW-SHOT POOL: {image, label, prob_T} ################################
        if (self.fewshot_pool is None) | (fewshot_pool is not None):
            self.fewshot_pool = fewshot_pool
            self.fewshot_pool.transforms[0] = self.distill_pool_config.get('aug_pseudolabel')
            if (not self.ddp) | (self.rank == self.master_rank):
                prob_T_pool = torch.empty(size=[0, self.num_classes], device=self.device)
                if self.filter.get('mode') in ['mean', 'quantile']:
                    confidence = [torch.empty(size=[0], device=self.device) for k in range(self.num_classes)]
                # COMPUTE TEACHER'S PREDICTION AND CONFIDENCE ##################
                for data in fewshot_pool.get_loader(batch_size=self.batch_size, shuffle=False):
                    x_real, y_real = data
                    x_real = x_real.to(self.device)
                    prob_T = self.blackbox(self.teacher(x_real))

                    prob_T_pool = torch.cat(tensors=[prob_T_pool, prob_T], dim=0)
                    if self.filter.get('mode') in ['mean', 'quantile']:
                        for k in range(self.num_classes):
                            confidence[k] = torch.cat(tensors=[confidence[k], prob_T[y_real == k, k]], dim=0)

                # COMPUTE THRESHOLDS ###########################################
                if self.filter.get('mode') != None:
                    self.threshold = torch.zeros(size=[self.num_classes], device=self.device)
                if self.filter.get('mode') == 'constant':
                    # mode = 'constant': Filter with a specified constant
                    self.threshold[:] = self.filter['value']
                elif self.filter.get('mode') == 'quantile':
                    # mode = 'quantile': Adaptive thresholds (default)
                    if self.filter.get('per_class') == True:
                        for k in range(self.num_classes):
                            self.threshold[k] = torch.quantile(confidence[k], q=self.filter['q'], interpolation='linear')
                    elif self.filter.get('per_class') == False:
                        confidence = torch.cat(tensors=confidence)
                        self.threshold[:] = torch.quantile(confidence, q=self.filter['q'], interpolation='linear')
                    self.threshold = torch.clamp(
                        input=self.threshold,
                        min=self.filter['min_confidence'],
                        max=self.filter['max_confidence'],
                    )
                    del confidence
                elif self.filter.get('mode') == 'mean':
                    # mean - Not a good fit for long tail distribution
                    if self.filter.get('per_class') == True:
                        for k in range(self.num_classes):
                            self.threshold[k] = confidence[k].mean(dim=0)
                    elif self.filter.get('per_class') == False:
                        confidence = torch.cat(tensors=confidence)
                        self.threshold[:] = confidence.mean(dim=0)
                    del confidence
            else:
                prob_T_pool = torch.empty(size=[len(self.fewshot_pool), self.num_classes], device=self.device)
                self.threshold = torch.empty(size=[self.num_classes], device=self.device)
            
            if self.ddp:
                dist.broadcast(tensor=prob_T_pool, src=self.master_rank)
                dist.broadcast(tensor=self.threshold, src=self.master_rank)
                self.train_metrics['count_T'].metric.update(prob_T_pool)
                self.train_metrics['query'].metric.update(prob_T_pool.shape[0])
            self.fewshot_pool.add_field(field=prob_T_pool, getitem=torch.from_numpy)

        # STAGE 1: TRAINING GENERATOR WITH FEW-SHOT DATA #######################
        if stage == 1:
            self.train_metrics['stage'].update(stage)
            self.fewshot_pool.transforms[0] = self.gan_pool_config.get('aug_gan')
            # Pool for Generation Stage: {image, label}
            self.fewshot_cache_loader = DataLoader(
                dataset=self.fewshot_pool,
                batch_size=len(self.fewshot_pool), # Cache and return the whole dataset in one go
                shuffle=False,
                num_workers=1,           # Since fewshot data is fixed, these settings allow efficient caching
                pin_memory=True,         # Since fewshot data is fixed, these settings allow efficient caching
                prefetch_factor=1,       # Since fewshot data is fixed, these settings allow efficient caching
                persistent_workers=True, # Since fewshot data is fixed, these settings allow efficient caching
            )
            fewshot_cache = next(iter(self.fewshot_cache_loader))
            self.gan_pool = TensorPool(tensors=[fewshot_cache[0], fewshot_cache[1]])
            self.candidates = [TensorPool(dims=[self.image_dim]) for k in range(self.num_classes)]
            self.train_metrics['pool_size'].update(len(self.gan_pool))

            if self.ddp:
                sampler = DistributedSampler(
                    dataset=self.gan_pool,
                    num_replicas=self.world_size,
                    rank=self.rank,
                )
            else:
                sampler = None
            
            trainloader=self.gan_pool.get_loader(
                batch_size=self.batch_size//self.world_size,
                sampler=sampler,
                shuffle=True if sampler is None else False,
                persistent_workers=False, # Since GAN Pool will be dynamically updated, these settings allow accurate data loading (No loading in advance, etc.)
                pin_memory=False,         # Since GAN Pool will be dynamically updated, these settings allow accurate data loading (No loading in advance, etc.)
            )
            self.train_batch = self.train_batch_1
            return super().training_loop(trainloader=trainloader, **kwargs)

        # Stage 2: DISTILLATION ################################################
        elif stage == 2:
            self.train_metrics['stage'].update(stage)
            self.train_metrics['count_T'].reset()
            self.train_metrics['count_T'].cumulate = True
            # Pool for Distillation Stage: {image, prob_T} #####################
            distill_fewshot_pool = NdArrayPool(
                fields=[self.fewshot_pool.fields[0], self.fewshot_pool.fields[2]],
                getitems=[self.fewshot_pool.getitems[0], self.fewshot_pool.getitems[2]],
                transforms=[self.distill_pool_config.get('aug_kd_fewshot'), self.fewshot_pool.transforms[2]],
            )
            # Add/create synthetic data ########################################
            if synth_data is None:
                # Synthesize data from scratch
                # Check synthetic data
                candidates = [
                    TensorPool(tensors=[
                        torch.zeros(size=[self.distill_pool_config.get('budget')//self.num_classes, *self.image_dim]),
                        torch.zeros(size=[self.distill_pool_config.get('budget')//self.num_classes, self.num_classes]),
                    ])
                    for k in range(self.num_classes)
                ]
                counts = torch.zeros(self.num_classes, dtype=torch.int64, device=self.device)

                if (not self.ddp) or (self.ddp & self.rank == self.master_rank):
                    pbar = tqdm.tqdm(total=self.distill_pool_config.get('budget'), desc='Synth. data', leave=False, unit='samples')
                while True:
                    # Generate a batch of synthetic images with 4x batch_size for speed
                    with torch.inference_mode():
                        z_synth = torch.normal(mean=0, std=1, size=[self.batch_size*4, self.latent_dim], device=self.device)
                        x_synth = self.generator(z_synth)
                        prob_T = self.blackbox(self.teacher(x_synth))
                        confidence, pseudolabel = prob_T.max(dim=1)
                        self.train_metrics['count_T'].update(prob_T)
                        self.train_metrics['query'].update(prob_T.shape[0])

                    for k in range(self.num_classes):
                        # Skip this class if have enough samples in balanced settings
                        if ((self.distill_pool_config.get('balanced_classes') == True)
                            & (counts[k] >= self.distill_pool_config.get('budget')//self.num_classes)):
                            continue

                        # Filter by class, and optionally threshold
                        mask = (pseudolabel == k)
                        if self.distill_pool_config.get('filter') == True:
                            mask = mask & (confidence > self.threshold[k])
                        x_synth_k = x_synth[mask]
                        prob_T_k = prob_T[mask]
                        if self.ddp:
                            x_synth_k = torch.cat(tensors=all_gather_nd(tensor=x_synth_k, world_size=self.world_size), dim=0)
                            prob_T_k = torch.cat(tensors=all_gather_nd(tensor=prob_T_k, world_size=self.world_size), dim=0)

                        # Selected amount (taking 'just enough' for balanced classes)
                        num_selected_k = x_synth_k.shape[0]
                        if self.distill_pool_config.get('balanced_classes') == True:
                            num_selected_k = min(
                                num_selected_k,
                                self.distill_pool_config.get('budget')//self.num_classes - counts[k],
                            )
                        
                        # Overwrite selected to pre-allocated slots
                        candidates[k].overwrite(
                            indices=torch.arange(start=counts[k], end=counts[k]+num_selected_k),
                            new_data=[x_synth_k[0:num_selected_k], prob_T_k[0:num_selected_k]]
                        )
                        counts[k] = counts[k] + num_selected_k
                    
                    # Update progress bar
                    if self.distill_pool_config.get('balanced_classes') == True:
                        num_selected = counts.clamp(min=None, max=self.distill_pool_config.get('budget')//self.num_classes)
                    num_selected = num_selected.sum(dim=0).item()
                        
                    if (not self.ddp) or (self.ddp & self.rank == self.master_rank):
                        pbar.update(num_selected - pbar.n)
                        lowest_k_count, lowest_k  = counts.min(dim=0)
                        pbar.set_postfix({'lowest_class': f'[{lowest_k.item()}: {lowest_k_count.item()}]'})

                    # Exit with enough samples
                    if num_selected >= self.distill_pool_config.get('budget'):
                        overflow = counts.sum(dim=0) - self.distill_pool_config.get('budget')
                        break
                
                # Add to distill set, cap by budget
                distill_generated_pool = TensorPool(
                    tensors=[torch.zeros(size=[0, *self.image_dim]), torch.zeros(size=[0, self.num_classes])],
                    transforms=[self.distill_pool_config.get('aug_kd_synth'), None],
                ) 
                for k in range(self.num_classes):
                    distill_generated_pool.add(candidates[k].fields)
                # Capping Distillation set size (last class is slightly affected here - for future fix)
                if overflow > 0:
                    distill_generated_pool.delete(torch.arange(start=-overflow, end=0))
            elif isinstance(synth_data, dict):
                # Add synthetic data from a dict
                x_synth = synth_data.get('x', None)
                prob_T = synth_data.get('y', None)
                if prob_T is None:
                    prob_T = self.blackbox(self.teacher(x_synth))
                self.distill_pool.add([x_synth, prob_T])
                self.train_metrics['count_T'].update(prob_T)
                self.train_metrics['query'].update(prob_T.shape[0])
            elif isinstance(synth_data, DataLoader):
                # Add synthetic data from a DataLoader
                for data in synth_data:
                    (x_synth,) = data
                    x_synth = x_synth.to(device=self.device)
                    prob_T = self.blackbox(self.teacher(x_synth))
                    self.distill_pool.add([x_synth, prob_T])
                    self.train_metrics['count_T'].update(prob_T)
                    self.train_metrics['query'].update(prob_T.shape[0])
            
            # Concat and shuffle fewshot and synthetic data
            self.distill_pool = ConcatDataset(datasets=[distill_fewshot_pool, distill_generated_pool])
            self.train_metrics['pool_size'].update(len(self.distill_pool))
            # Dataloader
            if self.ddp:
                sampler = DistributedSampler(
                    dataset=self.distill_pool,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                )
            else:
                sampler = None
            
            trainloader = DataLoader(
                dataset=self.distill_pool,
                batch_size=self.batch_size//self.world_size,
                sampler=sampler,
                shuffle=True if sampler is None else False,
            )

            # Train student on distill set
            self.train_batch = self.train_batch_2
            return super().training_loop(trainloader=trainloader, **kwargs)
        
    # SUPPORT FOR DDP ADDED - SHOULD BE BUG-FREE
    def gradient_penalty(self, x_real:Tensor, x_synth:Tensor) -> Tensor:
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
        return penalty# + 0 * mixed_scores[0]

    # SUPPORT FOR DDP ADDED - SHOULD BE BUG-FREE
    @staticmethod
    def blackbox(x:Tensor) -> Tensor:
        return x.clone().detach().softmax(dim=1)
    
    # SUPPORT FOR DDP ADDED - SHOULD BE BUG-FREE
    def hook_callbacks(self, callbacks:Optional[Sequence[Callback]]):
        super().hook_callbacks(callbacks=callbacks)
        # Register print format to progress bar
        if (not self.ddp) or (self.ddp & self.rank == self.master_rank):
            for cb in self.callbacks:
                if isinstance(cb, ProgressBar):
                    cb.register_format(format_dict={
                        'count_T':cb.format_multicounter,
                        'count_S':cb.format_multicounter,
                    })



def save_images(images:Tensor|np.ndarray, is_bhwc:bool=False, path='./logs/image.png', return_figax:bool=False, **kwargs):
    if images.dim() == 3:
        images = images.unsqueeze(dim=0)
    elif images.dim() != 4: 
        raise NotImplementedError
    
    if is_bhwc:
        # Convert to BCHW
        images = images.permute(0, 3, 1, 2)

    fig, ax = show_aug(images, **kwargs)
    fig.savefig(path)

    if return_figax:
        return fig, ax


# Getitem functions
def getitem_x_mnist(img:Tensor) -> Image:
    # torch.uint8 0-255
    return Image.fromarray(img, mode="L")
def getitem_x_svhn(img:np.ndarray) -> Image:
    # np.uint8 0-255
    return Image.fromarray(np.transpose(img, (1, 2, 0)))
def getitem_x_cifar(img:np.ndarray) -> Image:
    # np.uint8 0-255
    return Image.fromarray(img)
def getitem_x_imagenet(img:Tuple[str, str]) -> Image:
    # img: np.ndarray[path, label]
    return Image.open(img[0]).convert("RGB")
def getitem_y_mnist(target:Tensor|np.ndarray) -> int:
    # torch.int64 | np.int64 | str
    return int(target)
def find_ndarray_pool_getitems(dataset:str) -> Callable:
    if dataset in ['MNIST', 'FashionMNIST']:
        return getitem_x_mnist, getitem_y_mnist
    if dataset == 'SVHN':
        return getitem_x_svhn, getitem_y_mnist
    if dataset in ['CIFAR10', 'CIFAR100']:
        return getitem_x_cifar, getitem_y_mnist
    if dataset == 'TinyImageNet':
        return getitem_x_cifar, None
    if dataset == 'Imagenette':
        return getitem_x_imagenet, getitem_y_mnist


def expt_main(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    # SETUP CONFIG #############################################################
    metadata = METADATA[config['dataset']['name']]
    if world_size > 1:
        setup_process(rank=rank, world_size=world_size)
    if (rank == 0) | (rank is None):
        # print(f" {config['models']['distiller']['Class'].__name__} - "
        #     f"{config['models']['teacher']['Class'].__name__}(**{config['models']['teacher']['kwargs']}) - "
        #     f"{config['models']['student']['Class'].__name__}(**{config['models']['student']['kwargs']}) - "
        #     f"{config['dataset']['name']} - run {run} ".center(100,'#')
        # )
        pprint(config)

    # DATA #####################################################################
    dataset = get_dataset(
        name=config['dataset']['name'],
        **config['dataset']['kwargs'],
    )
    if config['dataset'].get('split_remap') is not None:
        for k, v in config['dataset']['split_remap'].items():
            dataset[v] = dataset.pop(k)
    dataset['train'] = make_fewshot(dataset=dataset['train'], **config['dataset']['fewshot_kwargs'])
    dataloader = get_dataloader(
        dataset=dataset,
        ddp=((world_size > 1) & config['dataloader']['ddp']),
        ddp_kwargs={
            'rank':rank,
            'world_size':world_size,
            **config['dataloader']['ddp_kwargs'],
        } if (world_size > 1) else None,
        batch_size={k:v//world_size for k, v in config['dataloader']['batch_size'].items()},
        shuffle=None if ((world_size > 1) & config['dataloader']['ddp']) else {'train':True, 'test':False},
        **config['dataloader']['kwargs'],
    )
    fewshot_pool = NdArrayPool(
        fields=[
            getattr(dataset['train'], metadata['attr_inputs']),
            getattr(dataset['train'], metadata['attr_targets']),
        ],
        getitems=[*find_ndarray_pool_getitems(config['dataset']['name'])],
    )

    # TEACHER ##################################################################
    model_T:nn.Module = config['models']['teacher']['Class'](**config['models']['teacher']['kwargs'])
    if config['models']['teacher'].get('name') is not None:
        model_T.name = config['models']['teacher']['name']
    trainer = ClassifierTrainer(model=model_T, device=rank, world_size=world_size)
    trainer.compile()
    model_T.load_state_dict(torch.load(
        f=config['models']['teacher']['pretrained_path'],
        map_location=trainer.device,
    ))
    trainer.evaluate(dataloader['test'])

    # MODELS ###################################################################
    model_G:nn.Module = config['models']['generator']['Class'](**config['models']['generator']['kwargs'])
    model_D:nn.Module = config['models']['discriminator']['Class'](**config['models']['discriminator']['kwargs'])
    model_S:nn.Module = config['models']['student']['Class'](**config['models']['student']['kwargs'])
    if config['models']['student'].get('name') is not None:
        model_S.name = config['models']['student']['name']

    distiller:DivBFKD = config['models']['distiller']['Class'](
        teacher=model_T,
        student=model_S,
        generator=model_G,
        discriminator=model_D,
        device=rank,
        world_size=world_size,
        **config['models']['distiller']['kwargs'],
    )
    opt_G = config['optim']['opt_G']['Class'](params=model_G.parameters(), **config['optim']['opt_G']['kwargs'])
    opt_D = config['optim']['opt_D']['Class'](params=model_D.parameters(), **config['optim']['opt_D']['kwargs'])
    opt_S = config['optim']['opt_S']['Class'](params=model_S.parameters(), **config['optim']['opt_S']['kwargs'])
    distiller.compile(
        opt_G=opt_G,
        opt_D=opt_D,
        opt_S=opt_S,
        **config['models']['distiller']['compile_kwargs'],
    )

    # SETUP TRAINING LOOP ######################################################
    callbacks = {1:[], 2:[]} # Indexed by training loop's stage
    if (rank == 0) | (rank is None):
        csv_logger = CSVLogger(
            filename=f"./logs/{config['dataset']['name']}/{distiller.name} - {model_T.name} - {model_S.name} - run {run}.csv",
            append=True,
        )
        gif_maker = GANGIFMaker(
            generator=model_G,
            filename=f"./logs/{config['dataset']['name']}/{distiller.name} - {model_T.name} - {model_S.name} - run {run}.gif",
            normalize=False,
            save_freq=config['optim']['num_epochs'][0]//50 if (config['optim']['num_epochs'][0]%50 == 0) else 1,
        )
        callbacks[1].extend([csv_logger, gif_maker])
        callbacks[2].append(csv_logger)
    if config['optim'].get('sch_S') is not None:
        sch_S = config['optim']['sch_S']['Class'](optimizer=opt_S, **config['optim']['sch_S']['kwargs'])
        sch_S_cb = SchedulerOnEpoch(on_epoch='train_end', scheduler=sch_S, key='lr_S')
        callbacks[1].append(sch_S_cb)
        callbacks[2].append(sch_S_cb)

    distiller.training_loop(
        stage=1,
        num_epochs=config['optim']['num_epochs'][0],
        fewshot_pool=fewshot_pool,
        valloader=dataloader['test'],
        callbacks=callbacks[1],
        val_freq=config['optim']['num_epochs'][0]-1,
    )
    if (config['optim']['save_G']) & ((rank == 0) | (rank is None)):
        # For datasets with extremely high number of classes, some classes 
        # might be very scarce and takes very long to accumulate enough 
        # samples (following the `balanced_classes` option). In case this
        # happens, the generator model is saved at the end of stage 1, so
        # stage 2 can be manually start again from this checkpoint.
        torch.save(obj=model_G.state_dict(), f=f'{csv_logger.filename[0:-4]} - model_G.pt')
    distiller.training_loop(
        stage=2,
        num_epochs=config['optim']['num_epochs'][1],
        valloader=dataloader['test'],
        callbacks=callbacks[2],
        start_epoch=config['optim']['num_epochs'][0],
        val_freq=config['optim']['num_epochs'][1]-1,
    )
    
    if world_size > 1:
        cleanup()


def expt_teacher(rank:int, world_size:int, config:dict, run:int=0):
    # SETUP CONFIG #############################################################
    if world_size > 1:
        setup_process(rank=rank, world_size=world_size)

    # DATA #####################################################################
    dataset = get_dataset(name=config['dataset']['name'], **config['dataset']['kwargs'])
    if config['dataset'].get('split_remap') is not None:
        for k, v in config['dataset']['split_remap'].items():
            dataset[v] = dataset.pop(k)
    dataloader = get_dataloader(
        dataset=dataset,
        ddp=((world_size > 1) & config['dataloader']['ddp']),
        ddp_kwargs={
            'rank':rank,
            'world_size':world_size,
            **config['dataloader']['ddp_kwargs'],
        } if (world_size > 1) else None,
        batch_size={k:v//world_size for k, v in config['dataloader']['batch_size'].items()},
        shuffle=None if ((world_size > 1) & config['dataloader']['ddp']) else {'train':True, 'test':False},
        **config['dataloader']['kwargs'],
    )

    # MODELS ###################################################################
    model_T:nn.Module = config['models']['teacher']['Class'](**config['models']['teacher']['kwargs'])
    if config['models']['teacher'].get('name') is not None:
        model_T.name = config['models']['teacher']['name']
    trainer:ClassifierTrainer = config['models']['trainer']['Class'](model=model_T, device=rank, world_size=world_size)
    opt_T = config['optim']['opt_T']['Class'](params=model_T.parameters(), **config['optim']['opt_T']['kwargs'])
    trainer.compile(opt=opt_T, **config['models']['trainer']['compile_kwargs'])

    # SETUP TRAINING LOOP ######################################################
    callbacks = []
    if (rank == 0) | (rank is None):
        csv_logger = CSVLogger(
            filename=os.path.join(config['metaconfig']['path'], f"{config['metaconfig']['name']} - run {run}.csv"),
            append=True,
        )
        best_callback = ModelCheckpoint(
            target=model_T,
            filepath=os.path.join(config['metaconfig']['path'], f"{config['metaconfig']['name']} - run {run}.pt"),
            monitor='val_acc',
            save_best_only=True,
        )
        callbacks.extend([csv_logger, best_callback])
    if config['optim'].get('sch_T') is not None:
        sch_T = config['optim']['sch_T']['Class'](optimizer=opt_T, **config['optim']['sch_T']['kwargs'])
        sch_T_cb = LearningRateSchedulerOnEpoch(scheduler=sch_T, log_key='lr')
        callbacks.append(sch_T_cb)
    
    inference_only = False
    if not inference_only:
        trainer.training_loop(
            trainloader=dataloader['train'],
            num_epochs=config['optim']['num_epochs'],
            valloader=dataloader['test'],
            callbacks=callbacks,
            val_freq=config['optim'].get('val_freq', 1),
        )
    
    model_T.load_state_dict(torch.load(
        f=best_callback.filepath,
        map_location=trainer.device,
        weights_only=True,
    ))
    trainer.evaluate(valloader=dataloader['test'])

    if world_size > 1:
        cleanup()


def expt_studentfull(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    # SETUP CONFIG #############################################################
    metadata = METADATA[config['dataset']['name']]
    if world_size > 1:
        setup_process(rank=rank, world_size=world_size)
    # if rank == 0:
    #     pprint(config)

    # DATA #####################################################################
    dataset = get_dataset(name=config['dataset']['name'], **config['dataset']['kwargs'])
    if config['dataset'].get('split_remap') is not None:
        for k, v in config['dataset']['split_remap'].items():
            dataset[v] = dataset.pop(k)
    dataloader = get_dataloader(
        dataset=dataset,
        ddp=((world_size > 1) & config['dataloader']['ddp']),
        ddp_kwargs={
            'rank':rank,
            'world_size':world_size,
            **config['dataloader']['ddp_kwargs'],
        } if (world_size > 1) else None,
        batch_size={k:v//world_size for k, v in config['dataloader']['batch_size'].items()},
        shuffle=None if ((world_size > 1) & config['dataloader']['ddp']) else {'train':True, 'test':False},
        **config['dataloader']['kwargs'],
    )

    # MODELS ###################################################################
    model_S:nn.Module = config['models']['student']['Class'](**config['models']['student']['kwargs'])
    if config['models']['student'].get('name') is not None:
        model_S.name = config['models']['student']['name']
    trainer = ClassifierTrainer(model=model_S, device=rank, world_size=world_size)
    opt_S = config['optim']['opt_S']['Class'](params=model_S.parameters(), **config['optim']['opt_S']['kwargs'])
    trainer.compile(opt=opt_S)

    # SETUP TRAINING LOOP ######################################################
    callbacks = []
    if (rank == 0) | (rank is None):
        csv_logger = CSVLogger(
            filename=f"./logs/{config['dataset']['name']}/{model_S.name} - run {run}.csv",
            append=True,
        )
        callbacks.append(csv_logger)
    if config['optim'].get('sch_S') is not None:
        sch_S = config['optim']['sch_S']['Class'](optimizer=opt_S, **config['optim']['sch_S']['kwargs'])
        sch_S_cb = SchedulerOnEpoch(on_epoch='train_end', scheduler=sch_S, key='lr')
        callbacks.append(sch_S_cb)
    trainer.training_loop(
        trainloader=dataloader['train'],
        num_epochs=config['optim']['num_epochs'],
        valloader=dataloader['test'],
        callbacks=callbacks,
        val_freq=config['optim']['num_epochs']-1,
    )

    if world_size > 1:
        cleanup()

def expt_studentalone(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    # SETUP CONFIG #############################################################
    metadata = METADATA[config['dataset']['name']]
    if world_size > 1:
        setup_process(rank=rank, world_size=world_size)
    # if (rank == 0) | (rank is None):
    #     pprint(config)

    # DATA #####################################################################
    dataset = get_dataset(name=config['dataset']['name'], **config['dataset']['kwargs'])
    if config['dataset'].get('split_remap') is not None:
        for k, v in config['dataset']['split_remap'].items():
            dataset[v] = dataset.pop(k)
    dataset['train'] = make_fewshot(dataset=dataset['train'], **config['dataset']['fewshot_kwargs'])
    dataloader = get_dataloader(
        dataset=dataset,
        ddp=((world_size > 1) & config['dataloader']['ddp']),
        ddp_kwargs={
            'rank':rank,
            'world_size':world_size,
            **config['dataloader']['ddp_kwargs'],
        } if (world_size > 1) else None,
        batch_size={k:v//world_size for k, v in config['dataloader']['batch_size'].items()},
        shuffle=None if ((world_size > 1) & config['dataloader']['ddp']) else {'train':True, 'test':False},
        **config['dataloader']['kwargs'],
    )

    # MODELS ###################################################################
    model_S:nn.Module = config['models']['student']['Class'](**config['models']['student']['kwargs'])
    if config['models']['student'].get('name') is not None:
        model_S.name = config['models']['student']['name']
    trainer = ClassifierTrainer(model=model_S, device=rank, world_size=world_size)
    opt_S = config['optim']['opt_S']['Class'](params=model_S.parameters(), **config['optim']['opt_S']['kwargs'])
    trainer.compile(opt=opt_S)

    # SETUP TRAINING LOOP ######################################################
    callbacks = []
    if (rank == 0) | (rank is None):
        csv_logger = CSVLogger(
            filename=f"./logs/{config['dataset']['name']}/{model_S.name} - run {run}.csv",
            append=True,
        )
        callbacks.append(csv_logger)
    if config['optim'].get('sch_S') is not None:
        sch_S = config['optim']['sch_S']['Class'](optimizer=opt_S, **config['optim']['sch_S']['kwargs'])
        sch_S_cb = SchedulerOnEpoch(on_epoch='train_end', scheduler=sch_S, key='lr')
        callbacks.append(sch_S_cb)
    trainer.training_loop(
        trainloader=dataloader['train'],
        num_epochs=config['optim']['num_epochs'],
        valloader=dataloader['test'],
        callbacks=callbacks,
        val_freq=config['optim']['num_epochs']-1,
    )

    if world_size > 1:
        cleanup()

def expt_standardkd(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    # SETUP CONFIG #############################################################
    metadata = METADATA[config['dataset']['name']]
    if world_size > 1:
        setup_process(rank=rank, world_size=world_size)
    # if rank == 0:
    #     pprint(config)

    # DATA #####################################################################
    dataset = get_dataset(name=config['dataset']['name'], **config['dataset']['kwargs'])
    if config['dataset'].get('split_remap') is not None:
        for k, v in config['dataset']['split_remap'].items():
            dataloader[v] = dataloader.pop(k)
    dataset['train'] = make_fewshot(dataset=dataset['train'], **config['dataset']['fewshot_kwargs'])
    dataloader = get_dataloader(
        dataset=dataset,
        ddp=((world_size > 1) & config['dataloader']['ddp']),
        ddp_kwargs={
            'rank':rank,
            'world_size':world_size,
            **config['dataloader']['ddp_kwargs'],
        } if (world_size > 1) else None,
        batch_size={k:v//world_size for k, v in config['dataloader']['batch_size'].items()},
        shuffle=None if ((world_size > 1) & config['dataloader']['ddp']) else {'train':True, 'test':False},
        **config['dataloader']['kwargs'],
    )

    # TEACHER ##################################################################
    model_T:nn.Module = config['models']['teacher']['Class'](**config['models']['teacher']['kwargs'])
    if config['models']['teacher'].get('name') is not None:
        model_T.name = config['models']['teacher']['name']

    trainer = ClassifierTrainer(model=model_T, device=rank, world_size=world_size)
    trainer.compile()
    model_T.load_state_dict(torch.load(
        f=config['models']['teacher']['pretrained_path'],
        map_location=trainer.device,
    ))
    trainer.evaluate(dataloader['test'])

    # MODELS ###################################################################
    model_S:nn.Module = config['models']['student']['Class'](**config['models']['student']['kwargs'])
    if config['models']['student'].get('name') is not None:
        model_S.name = config['models']['student']['name']

    distiller:HintonDistiller = config['models']['distiller']['Class'](
        teacher=model_T,
        student=model_S,
        image_dim=config['dataset']['image_dim'],
        device=rank,
        world_size=world_size,
    )

    # OPTIMIZATION #############################################################
    opt_S = config['optim']['opt_S']['Class'](params=model_S.parameters(), **config['optim']['opt_S']['kwargs'])
    distiller.compile(opt=opt_S, **config['models']['distiller']['compile_kwargs'])

    # SETUP TRAINING LOOP ######################################################
    callbacks = []
    if (rank == 0) | (rank is None):
        csv_logger = CSVLogger(
            filename=f"./logs/{config['dataset']['name']}/{distiller.name} - {model_T.name} - {model_S.name} - run {run}.csv",
            append=True,
        )
        callbacks.append(csv_logger)
    if config['optim'].get('sch_S') is not None:
        sch_S = config['optim']['sch_S']['Class'](optimizer=opt_S, **config['optim']['sch_S']['kwargs'])
        sch_S_cb = SchedulerOnEpoch(on_epoch='train_end', scheduler=sch_S, key='lr_S')
        callbacks.append(sch_S_cb)

    distiller.training_loop(
        trainloader=dataloader['train'],
        num_epochs=config['optim']['num_epochs'],
        valloader=dataloader['test'],
        callbacks=callbacks,
        val_freq=config['optim']['num_epochs']-1,
    )

    if world_size > 1:
        cleanup()

def expt_abla_components(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    return NotImplementedError


def expt_wgan(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    if world_size > 1:
        setup_process(rank=rank, world_size=world_size)
    # if rank == 0:
    #     pprint(config)

    # DATA #####################################################################
    dataset = get_dataset(
        name=config['dataset']['name'],
        **config['dataset']['kwargs'],
    )
    dataset['train'] = make_fewshot(
        dataset=dataset['train'],
        **config['dataset']['fewshot_kwargs'],
    )
    dataloader = get_dataloader(
        dataset=dataset,
        ddp=((world_size > 1) & config['dataloader']['ddp']),
        ddp_kwargs={
            'rank':rank,
            'world_size':world_size,
            **config['dataloader']['ddp_kwargs'],
        } if (world_size > 1) else None,
        batch_size={k:v//world_size for k, v in config['dataloader']['batch_size'].items()},
        **config['dataloader']['kwargs'],
    )
    if config['dataset'].get('split_remap') is not None:
        for k, v in config['dataset']['split_remap'].items():
            dataloader[v] = dataloader.pop(k)

    # MODELS ###################################################################
    model_G:nn.Module = config['models']['generator']['Class'](**config['models']['generator']['kwargs'])
    model_C:nn.Module = config['models']['discriminator']['Class'](**config['models']['discriminator']['kwargs'])
    wgan:WGAN_GP = config['models']['GAN']['Class'](
        generator=model_G,
        critic=model_C,
        device=rank,
    )

    opt_G = config['optim']['opt_G']['Class'](params=model_G.parameters(), **config['optim']['opt_G']['kwargs'])
    opt_C = config['optim']['opt_C']['Class'](params=model_C.parameters(), **config['optim']['opt_C']['kwargs'])
    wgan.compile(opt_G=opt_G, opt_C=opt_C, **config['models']['GAN']['compile_kwargs'])

    # SETUP TRAINING LOOP ######################################################
    callbacks = []
    if (rank == 0) | (rank is None):
        csv_logger = CSVLogger(filename=f"./logs/{config['dataset']['name']}/{wgan.name} - run {run}.csv", append=True)
        gif_maker = GANGIFMaker(
            generator=model_G,
            filename=f"./logs/{config['dataset']['name']}/{wgan.name} - run {run}.gif",
            normalize=False,
            save_freq=config['optim']['num_epochs']//50,
        )
        callbacks.extend([csv_logger, gif_maker])
    wgan.training_loop(
        trainloader=dataloader['train'],
        num_epochs=config['optim']['num_epochs'],
        valloader=dataloader['test'],
        callbacks=callbacks,
    )


def mixup(
    x:Tensor, y:Tensor, num_mixes:int, num_classes:Optional[int]=None, hard_label:bool=True,
    mix_distr = torch.distributions.beta.Beta(concentration1=0.2, concentration0=0.2),
    threshold:float=0,
) -> Tuple[Tensor, Tensor, Tensor]:
    num_samples = x.shape[0]
    if hard_label == True:
        if num_classes is None:
            num_classes = y.max()
        y = torch.nn.functional.one_hot(y, num_classes=num_classes)
    
    idx_left = torch.randint(low=0, high=num_samples, size=[num_mixes])
    # Right must not be duplicated with Left
    idx_right = idx_left + torch.randint(low=1, high=num_samples-1, size=[num_mixes])
    idx_right = idx_right % num_samples
    # Mixup coefficients
    mix:Tensor = mix_distr.sample(sample_shape=[num_mixes])
    boundaries = ((mix < threshold) | (mix > 1 - threshold)).nonzero().squeeze(dim=1)
    while boundaries.shape[0] > 0:
        mix[boundaries] = mix_distr.sample(sample_shape=[boundaries.shape[0]])
        boundaries = ((mix < threshold) | (mix > 1 - threshold)).nonzero().squeeze(dim=1)
    # mix = mix.round(decimals=2)
    mix_x = mix.view(num_mixes, *[1]*(x.dim() - 1))
    mix_y = mix.view(num_mixes, *[1]*(y.dim() - 1))

    x_mix = mix_x*x[idx_left] + (1 - mix_x)*x[idx_right]
    y_mix = mix_y*y[idx_left] + (1 - mix_y)*y[idx_right]
    # np.save('./tests/x_mix.npy', x_mix.numpy())
    return x_mix, y_mix, mix

def extract_features(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0) -> Tuple[Tensor, Tensor, Sequence[str]]:
    # DATA #####################################################################
    dataset = get_dataset(name=config['dataset']['name'], **config['dataset']['kwargs'])
    if config['dataset'].get('split_remap') is not None:
        for k, v in config['dataset']['split_remap'].items():
            dataset[v] = dataset.pop(k)
    dataset['fewshot'] = make_fewshot(dataset=dataset['train'], **config['dataset']['fewshot_kwargs'])
    dataloader = get_dataloader(
        dataset=dataset,
        ddp=((world_size > 1) & config['dataloader']['ddp']),
        ddp_kwargs={
            'rank':rank,
            'world_size':world_size,
            **config['dataloader']['ddp_kwargs'],
        } if (world_size > 1) else None,
        batch_size={k:v//world_size for k, v in config['dataloader']['batch_size'].items()},
        shuffle=None if ((world_size > 1) & config['dataloader']['ddp']) else {'fewshot':True, 'train':True, 'test':False},
        **config['dataloader']['kwargs'],
    )

    # MODELS ###################################################################
    model_T:nn.Module = config['models']['teacher']['Class'](**config['models']['teacher']['kwargs'])
    if config['models']['teacher'].get('name') is not None:
        model_T.name = config['models']['teacher']['name']
    trainer = ClassifierTrainer(model=model_T, device=rank, world_size=world_size)
    trainer.compile()
    model_T.load_state_dict(torch.load(
        f=config['models']['teacher']['pretrained_path'],
        map_location=trainer.device,
    ))
    trainer.evaluate(dataloader['test'])
    extractor_T = IntermediateFeatureExtractor(
        model=model_T,
        out_layers={'feature':getattr(model_T, config['models']['teacher']['feature_layer'])},
    )

    # EXTRACT FEATURES #########################################################
    features:dict[str, Tensor] = {}
    if config['embed']['toggle_pool']['trainfull']:
        # Full training set {x, y, h, y^}
        num_samples = len(dataset['train'])
        pool_trainfull = TensorPool(tensors=[
            torch.zeros(size=[num_samples, *config['dataset']['image_dim']], dtype=torch.float),
            torch.zeros(size=[num_samples], dtype=torch.int64),
            torch.zeros(size=[num_samples, *config['models']['teacher']['feature_dim']], dtype=torch.float),
            torch.zeros(size=[num_samples, config['dataset']['num_classes']], dtype=torch.float),
        ])
        counter = 0
        for data in tqdm.tqdm(dataloader['train'], desc=f"pool_trainfull ({len(dataset['train'])})", leave=True, unit='batches'):
            x, y = data
            x = x.to(device=extractor_T.device)
            batch_size = x.shape[0]
            feature_dict, pred = extractor_T(x)
            feature = feature_dict['out']['feature']
            pool_trainfull.overwrite(
                indices=torch.arange(counter, counter+batch_size),
                new_data=[x, y, feature, pred],
            )
            counter = counter + batch_size
        features.update({'trainfull': pool_trainfull.fields[2].clone()})
    if True:
        # Few-shot set {x, y, h, y^}
        # x_train_few = torch.tensor(np.load(file='./tests/fsbbt2_synthetic/abla_x_fewshot_2k.npy'))
        # y_train_few = torch.tensor(np.load(file='./tests/fsbbt2_synthetic/abla_y_fewshot_2k.npy')).argmax(dim=1)
        num_samples = len(dataset['fewshot'])
        pool_trainfew = TensorPool(tensors=[
            torch.zeros(size=[num_samples, *config['dataset']['image_dim']], dtype=torch.float),
            torch.zeros(size=[num_samples], dtype=torch.int64),
            torch.zeros(size=[num_samples, *config['models']['teacher']['feature_dim']], dtype=torch.float),
            torch.zeros(size=[num_samples, config['dataset']['num_classes']], dtype=torch.float),
        ])
        counter = 0
        for data in tqdm.tqdm(dataloader['fewshot'], desc=f"pool_trainfew ({len(dataset['fewshot'])})", leave=True, unit='batches'):
            x, y = data
            x = x.to(device=extractor_T.device)
            batch_size = x.shape[0]
            feature_dict, pred = extractor_T(x)
            feature = feature_dict['out']['feature']
            pool_trainfew.overwrite(
                indices=torch.arange(counter, counter+batch_size),
                new_data=[x, y, feature, pred],
            )
            counter = counter + batch_size
        if config['embed']['toggle_pool']['trainfew']:
            features.update({'trainfew': pool_trainfew.fields[2].clone()})
    if (config['embed']['toggle_pool']['mixupfew'] & config['embed']['toggle_pool']['trainfew']):
        # MixUp {x, y_mix, h, y^}
        num_samples = config['dataset']['mixup_kwargs']['num_samples']
        batch_size = config['dataset']['mixup_kwargs']['batch_size']
        pool_mixupfew = TensorPool(tensors=[
            torch.zeros(size=[num_samples, *config['dataset']['image_dim']], dtype=torch.float),
            torch.zeros(size=[num_samples, config['dataset']['num_classes']], dtype=torch.float),
            torch.zeros(size=[num_samples, *config['models']['teacher']['feature_dim']], dtype=torch.float),
            torch.zeros(size=[num_samples, config['dataset']['num_classes']], dtype=torch.float),
        ])
        counter = 0
        for i in tqdm.trange(num_samples//batch_size, desc=f"pool_mixupfew ({num_samples})", leave=True, unit='batches'):
            x, y, mix = mixup(
                x=pool_trainfew.fields[0],
                y=pool_trainfew.fields[1],
                num_mixes=config['dataset']['mixup_kwargs']['batch_size'],
                num_classes=config['dataset']['num_classes'],
                hard_label=True,
                mix_distr=torch.distributions.uniform.Uniform(low=0.1, high=0.9),
                threshold=0,
            )
            x = x.to(extractor_T.device)
            feature_dict, pred = extractor_T(x)
            feature = feature_dict['out']['feature']
            pool_mixupfew.overwrite(
                indices=torch.arange(counter, counter+batch_size),
                new_data=[x, y, feature, pred],
            )
            counter = counter + batch_size
        features.update({'mixupfew': pool_mixupfew.fields[2].clone()})
    if config['embed']['toggle_pool']['cvae']:
        # MixUp {x, y, h, y^}
        decoder = config['models']['cvae']['Class'](**config['models']['cvae']['kwargs'])
        decoder.compile()

        # torch.load(f='./tests/fsbbt2_synthetic/x_cvae_u_2k.pt')
        num_samples = config['dataset']['cvae_kwargs']['num_samples']
        batch_size = config['dataset']['cvae_kwargs']['batch_size']
        pool_cvae = TensorPool(tensors=[
            torch.zeros(size=[num_samples, *config['dataset']['image_dim']], dtype=torch.float),
            torch.zeros(size=[num_samples, config['dataset']['num_classes']], dtype=torch.float),
            torch.zeros(size=[num_samples, *config['models']['teacher']['feature_dim']], dtype=torch.float),
            torch.zeros(size=[num_samples, config['dataset']['num_classes']], dtype=torch.float),
        ])
        uniform = torch.distributions.uniform.Uniform(low=-3, high=3)
        counter = 0
        for i in tqdm.trange(num_samples//batch_size, desc=f"pool_cvae ({num_samples})", leave=True, unit='batches'):
            z_latent = uniform.sample(sample_shape=[batch_size, config['models']['cvae']['latent_dim']])
            z_k = nn.functional.one_hot(
                input=torch.randint(low=0, high=config['dataset']['num_classes'], size=[batch_size]),
                num_classes=config['dataset']['num_classes'],
            ).to(dtype=torch.float)
            z = torch.cat(tensors=[z_latent, z_k], dim=1)
            x = torch.from_numpy(decoder(z.numpy()).numpy())
            x = x.view(batch_size, 32, 32, 3)
            x = x.permute(0, 3, 1, 2)
            x = x.to(extractor_T.device)
            feature_dict, pred = extractor_T(x)
            feature = feature_dict['out']['feature']

            pool_cvae.overwrite(
                indices=torch.arange(counter, counter+batch_size),
                new_data=[x, y, feature, pred],
            )
            counter = counter + batch_size
        features.update({'cvae': pool_cvae.fields[2].clone()})
    if config['embed']['toggle_pool']['wgan']:
        # WGAN {x, y=-1, h, y^}
        model_G = config['models']['generator']['Class'](**config['models']['generator']['kwargs']).to(extractor_T.device)
        model_G.load_state_dict(torch.load(
            f=config['models']['generator']['pretrained_path_wgan'],
            map_location=extractor_T.device,
        ))
        num_samples = config['dataset']['wgan_kwargs']['num_samples']
        batch_size = config['dataset']['wgan_kwargs']['batch_size']
        pool_wgan = TensorPool(tensors=[
            torch.zeros(size=[num_samples, *config['dataset']['image_dim']], dtype=torch.float),
            torch.zeros(size=[num_samples], dtype=torch.int64),
            torch.zeros(size=[num_samples, *config['models']['teacher']['feature_dim']], dtype=torch.float),
            torch.zeros(size=[num_samples, config['dataset']['num_classes']], dtype=torch.float),
        ])
        counter = 0
        for i in tqdm.trange(num_samples//batch_size, desc=f"pool_wgan ({num_samples})", leave=True, unit='batches'):
            z = torch.normal(mean=0, std=1, size=[config['dataset']['wgan_kwargs']['batch_size'], config['models']['generator']['kwargs']['latent_dim']], device=extractor_T.device)
            x = model_G(z)
            feature_dict, pred = extractor_T(x)
            feature = feature_dict['out']['feature']
            y = -torch.ones(size=[batch_size], dtype=torch.int64)
            pool_wgan.overwrite(
                indices=torch.arange(counter, counter+batch_size),
                new_data=[x, y, feature, pred],
            )
            counter = counter + batch_size
        features.update({'wgan': pool_wgan.fields[2].clone()})
    if (config['embed']['toggle_pool']['divbfkd']
        | config['embed']['toggle_pool']['divbfkd-unfiltered']
        | config['embed']['toggle_pool']['fullkd']):
        # DivBFKD {x, y=-1, h, y^}
        model_G = config['models']['generator']['Class'](**config['models']['generator']['kwargs']).to(trainer.device)
        model_G.load_state_dict(torch.load(
            f=config['models']['generator']['pretrained_path_divbfkd'],
            map_location=trainer.device,
        ))
        num_samples = config['dataset']['divbfkd_kwargs']['num_samples']
        batch_size = config['dataset']['divbfkd_kwargs']['batch_size']
        pool_divbfkd = TensorPool(tensors=[
            torch.zeros(size=[num_samples, *config['dataset']['image_dim']], dtype=torch.float),
            torch.zeros(size=[num_samples], dtype=torch.int64),
            torch.zeros(size=[num_samples, *config['models']['teacher']['feature_dim']], dtype=torch.float),
            torch.zeros(size=[num_samples, config['dataset']['num_classes']], dtype=torch.float),
        ])
        counter = 0
        for i in tqdm.trange(num_samples//batch_size, desc=f"pool_divbfkd ({num_samples})", leave=True, unit='batches'):
            z = torch.normal(mean=0, std=1, size=[batch_size, config['models']['generator']['kwargs']['latent_dim']], device=extractor_T.device)
            x = model_G(z)
            feature_dict, pred = extractor_T(x)
            feature = feature_dict['out']['feature']
            y = -torch.ones(size=[batch_size], dtype=torch.int64)
            pool_divbfkd.overwrite(
                indices=torch.arange(counter, counter+batch_size),
                new_data=[x, y, feature, pred],
            )
            counter = counter + batch_size

        if config['embed']['toggle_pool']['divbfkd-unfiltered']:
            features.update({'divbfkd-unfiltered': pool_divbfkd.fields[2].clone()})
        # Adaptive thresholds (default)
        threshold = torch.zeros(size=[config['dataset']['num_classes']])
        for k in range(config['dataset']['num_classes']):
            confidence_k = pool_trainfew.fields[3].softmax(dim=1)[pool_trainfew.fields[1]==k, k]
            threshold[k] = torch.quantile(confidence_k, q=0.1, interpolation='linear').clamp(0.7, 0.99)
        print(f'threshold: {threshold.numpy()}')
        # Filter
        selected = []
        confidence, pseudolabel = pool_divbfkd.fields[3].softmax(dim=1).max(dim=1)
        for k in range(config['dataset']['num_classes']):
            selected_k = ((pseudolabel == k) & (confidence >= threshold[k]))
            selected.append(selected_k.nonzero().squeeze(dim=1))
        selected = torch.cat(tensors=selected, dim=0)
        pool_divbfkd.slice(selected)
        pool_divbfkd.shuffle()
        if config['embed']['toggle_pool']['divbfkd']:
            features.update({'divbfkd': pool_divbfkd.fields[2].clone()})

        if config['embed']['toggle_pool']['fullkd']:
            features.update({'fullkd': torch.concat(tensors=[features['trainfew'], features['divbfkd-unfiltered']], dim=0).clone()})

    # EMBED ####################################################################
    x_all = torch.concat(tensors=[tensor for k, tensor in features.items()], dim=0)
    y_all = torch.concat(tensors=[i*torch.ones(size=[tensor.shape[0]], dtype=torch.int64) for i, (k, tensor) in enumerate(features.items())], dim=0)
    labels = [*features.keys()]
    print(f"Data: {[(label, (y_all == k).sum().item()) for k, label in enumerate(labels)]}")

    print(f"Begin embedding with {config['embed']['Class'].__name__}...")
    x_embed = config['embed']['Class'](**config['embed']['kwargs']).fit_transform(nn.Flatten()(x_all))
    print(f"Finish embedding.")

    # METRICS ##################################################################
    if config['metrics']['alpha_precision']:
        from models.distillers.alpha_precision import compute_metrics

        print(f'{labels[0]}/{labels[1]}')
        results = compute_metrics(
            X=torch.from_numpy(x_embed[(y_all==0).nonzero().squeeze(dim=1)]),
            Y=torch.from_numpy(x_embed[(y_all==1).nonzero().squeeze(dim=1)]),
        )[0]
        torch.save(obj=results, f=f"./logs/{config['dataset']['name']}/alpha_precision_{labels[0]}_{labels[1]}.pt")

        print(f'{labels[0]}/{labels[2]}')
        results = compute_metrics(
            X=torch.from_numpy(x_embed[(y_all==0).nonzero().squeeze(dim=1)]),
            Y=torch.from_numpy(x_embed[(y_all==2).nonzero().squeeze(dim=1)]),
        )[0]
        torch.save(obj=results, f=f"./logs/{config['dataset']['name']}/alpha_precision_{labels[0]}_{labels[2]}.pt")

        print(f'{labels[0]}/{labels[3]}')
        results = compute_metrics(
            X=torch.from_numpy(x_embed[(y_all==0).nonzero().squeeze(dim=1)]),
            Y=torch.from_numpy(x_embed[(y_all==3).nonzero().squeeze(dim=1)]),
        )[0]
        torch.save(obj=results, f=f"./logs/{config['dataset']['name']}/alpha_precision_{labels[0]}_{labels[3]}.pt")

        print(f'{labels[0]}/{labels[4]}')
        results = compute_metrics(
            X=torch.from_numpy(x_embed[(y_all==0).nonzero().squeeze(dim=1)]),
            Y=torch.from_numpy(x_embed[(y_all==4).nonzero().squeeze(dim=1)]),
        )[0]
        torch.save(obj=results, f=f"./logs/{config['dataset']['name']}/alpha_precision_{labels[0]}_{labels[4]}.pt")

        print(f'{labels[0]}/{labels[5]}')
        results = compute_metrics(
            X=torch.from_numpy(x_embed[(y_all==0).nonzero().squeeze(dim=1)]),
            Y=torch.from_numpy(x_embed[(y_all==5).nonzero().squeeze(dim=1)]),
        )[0]
        torch.save(obj=results, f=f"./logs/{config['dataset']['name']}/alpha_precision_{labels[0]}_{labels[5]}.pt")


    return torch.from_numpy(x_embed), y_all, labels


def vizembed2d(
    rank:int, world_size:int, config:dict, run:int=0, var:int|float=0,
    x_embed:Optional[Tensor]=None,
    y_all:Optional[Tensor]=None,
    labels:Optional[Sequence[str]]=None,
):
    # torch.save(
    #     obj={'x_all':x_all, 'x_embed':x_embed, 'y_all':y_all},
    #     f=f"./tests/fsbbt2_synthetic/embedding_{config['embed']['Class'].__name__}.pt",
    # )
    # VISUALIZE ################################################################
    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, constrained_layout=True, figsize=(16, 8))
    # fig.suptitle(f"Distributions of {config['dataset']['name']} data on 2D embedding space")
    for k, label in enumerate(labels):
        alpha = 0.12 if (label == 'trainfull') else None
        ax[0, 0].scatter(
            x_embed[y_all==k, 0], x_embed[y_all==k, 1],
            color=config['viz']['colors'][k],
            alpha=alpha,
            # label=label,
            # label=f'{label} ({(y_all==k).sum().item()})',
            **config['viz']['kwargs'],
        )
        
    from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', label="Teacher's training images", markerfacecolor='tab:red', markersize=15),
    #     Line2D([0], [0], marker='o', color='w', label="Few-shot images", markerfacecolor='black', markersize=15),
    #     Line2D([0], [0], marker='o', color='w', label="High-confidence images", markerfacecolor='tab:blue', markersize=15),
    #     Line2D([0], [0], marker='o', color='w', label='Distillation images', markerfacecolor='yellow', markersize=15),
    # ]
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label="Teacher's training images (Unseen)", markerfacecolor='tab:red', markersize=17),
        Line2D([0], [0], marker='o', color='w', label="Few-shot images", markerfacecolor='black', markersize=17),
        Line2D([0], [0], marker='o', color='w', label="Our synthetic images", markerfacecolor='tab:blue', markersize=17),
    ]

    ax[0, 0].legend(handles=legend_elements, fontsize="22")
    fig.savefig(
        fname=config['viz']['save_path'],
        dpi=1000, format=None, metadata=None,
        bbox_inches='tight', pad_inches=0,
        facecolor='auto', edgecolor='auto',
        backend=None
    )

    # Save as BHWC for Dang
    # np.save('./tests/fsbbt2_synthetic/X_wgan_2k.npy', pool_wgan.pool[0][0:2000].permute(0, 2, 3, 1).numpy())
    # np.save('./tests/fsbbt2_synthetic/X_divbfkd_2k.npy', pool_divbfkd.pool[0][0:2000].permute(0, 2, 3, 1).numpy())
    # np.save('./tests/fsbbt2_synthetic/X_wgan_40k.npy', pool_wgan.pool[0].permute(0, 2, 3, 1).numpy())
    # np.save('./tests/fsbbt2_synthetic/X_divbfkd_40k.npy', pool_divbfkd.pool[0].permute(0, 2, 3, 1).numpy())

def vizembed1d(
    rank:int, world_size:int, config:dict, run:int=0, var:int|float=0,
    x_embed:Optional[Tensor]=None,
    y_all:Optional[Tensor]=None,
    labels:Optional[Sequence[str]]=None,
):
    from sklearn.neighbors import KernelDensity

    p2p = x_embed.max() - x_embed.min()
    x_plot = torch.linspace(start=x_embed.min()-0.05*p2p, end=x_embed.max()+0.05*p2p, steps=1000).unsqueeze(dim=1)
    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, constrained_layout=True, figsize=(10, 6))
    # fig.suptitle(f"Probability density function of {config['dataset']['name']} data on 1D embedding space")
    for k, label in enumerate(labels):
        kde = KernelDensity().fit(x_embed[(y_all==k).nonzero().squeeze(dim=1), :])
        log_dens = torch.from_numpy(kde.score_samples(x_plot))
        ax[0, 0].fill_between(
            x_plot[:, 0], log_dens.exp(), 0,
            color=config['viz']['colors'][k],
            edgecolor=config['viz']['colors'][k],
            label=label,
            # label=f'{label} ({(y_all==k).sum().item()})',
            **config['viz']['kwargs'],
        )
    ax[0, 0].legend()
    ax[0, 0].set(
        ylim=[0, ax[0, 0].get_ylim()[1]],
        # xlabel='embedding',
        # ylabel='density',
    )

    fig.savefig(
        fname=config['viz']['save_path'],
        dpi='figure', format=None, metadata=None,
        bbox_inches='tight', pad_inches=0,
        facecolor='auto', edgecolor='auto',
        backend=None
    )

def expt_vizembed2d(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    x_embed, y_all, labels = extract_features(rank=rank, world_size=world_size, config=config, run=run, var=var)
    vizembed2d(
        rank=rank, world_size=world_size, config=config, run=run, var=var,
        x_embed=x_embed, y_all=y_all, labels=labels,
    )
def expt_vizembed1d(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    x_embed, y_all, labels = extract_features(rank=rank, world_size=world_size, config=config, run=run, var=var)
    vizembed1d(
        rank=rank, world_size=world_size, config=config, run=run, var=var,
        x_embed=x_embed, y_all=y_all, labels=labels,
    )


def parse_config_path(config:str, ext:str='yaml') -> str:
    config_splits = config.split('-')
    return f'./configs/selfboosting/{config_splits[0]}/{config}.{ext}'


def parse_expt(base_config:dict, custom_args:dict) -> Callable:
    config = deepcopy(base_config)
    
    if config['metaconfig']['expt'] == 'teacher':
        expt = expt_teacher
        keys_avail = ['num_epochs']
        for k, v in custom_args.items():
            assert k in keys_avail, f'Unknown key: {k}'
            if k == 'num_epochs':
                config['optim']['num_epoch'] = v
        return expt, config

    elif config['metaconfig']['expt'] == 'main':
        expt = expt_distillcontr
        keys_avail = [
            'checkpoint', 'num_generations', 'num_epochs', 'num_samples', 'tol_mc',
            'coeff_oh', 'coeff_ie', 'coeff_dc', 'coeff_ct', 'reduce',
            'p_bswap', 'beta', 'lr_X', 'wandb'
        ]
        for k, v in custom_args.items():
            assert k in keys_avail, f'Unknown key: {k}'
            if k == 'checkpoint':
                checkpoint_parsed:Sequence[str] = v.split('-')
                config['optim']['checkpoint'] = {
                    'g': int(checkpoint_parsed[0]),
                    'stage': checkpoint_parsed[1] if len(checkpoint_parsed) >= 2 else None,
                }
            if k == 'num_generations':
                config['optim']['num_generations'] = v
            elif k == 'num_epochs':
                config['optim']['num_epochs_distill'] = v
                config['optim']['num_epochs_synth'] = v
            elif k == 'num_samples':
                config['dataset']['pseudosamples']['noise']['compose_kwargs']['num_samples'] = v
                config['dataset']['pseudosamples']['optimized']['compose_kwargs']['num_samples'] = v
                config['models']['synthesizer']['compile_kwargs']['num_samples'] = v
            elif k == 'tol_mc':
                config['dataset']['pseudosamples']['noise']['compose_kwargs']['tolerance_missing_class'] = v
            elif k == 'coeff_oh':
                config['models']['synthesizer']['compile_kwargs']['coeff_oh'] = v
                config['models']['synthesizer']['compile_kwargs']['loss_fn_onehot'] = True
            elif k == 'coeff_ie':
                config['models']['synthesizer']['compile_kwargs']['coeff_ie'] = v
                config['models']['synthesizer']['compile_kwargs']['loss_fn_info_entropy'] = True
            elif k == 'coeff_dc':
                config['models']['synthesizer']['compile_kwargs']['coeff_dc'] = v
                config['models']['synthesizer']['compile_kwargs']['loss_fn_decay'] = True
            elif k == 'coeff_ct':
                config['models']['synthesizer']['compile_kwargs']['coeff_ct'] = v
                config['models']['synthesizer']['compile_kwargs']['loss_fn_contrast'] = True
            elif k == 'reduce':
                import re
                def reparse_projection_head(string:str, reduce:float) -> str:
                    numbers = re.findall(r'\d+', string)
                    if len(numbers) < 3:
                        return string  # Ensure there are at least 3 numbers
                    
                    numbers[0] = str(round(int(numbers[0])*reduce))  # Halve first argument
                    numbers[1] = str(round(int(numbers[1])*reduce))  # Halve second argument
                    
                    # VERY adhoc bugfix for AlexNet reduce = 0.1, 0.2
                    # Please make separate settings with upload code
                    if numbers[0] == "115":
                        numbers[0] = "117"
                        numbers[1] = "117"
                    elif numbers[0] == "230":
                        numbers[0] = "234"
                        numbers[1] = "234"
                    
                    num_iter = iter(numbers)
                    out = re.sub(r'\d+', lambda _: next(num_iter), string)
                    
                    return out
                print(config['models']['synthesizer']['kwargs']['projector'])
                print(v, reparse_projection_head(config['models']['synthesizer']['kwargs']['projector'], v))
                config['models']['synthesizer']['kwargs']['projector'] = reparse_projection_head(config['models']['synthesizer']['kwargs']['projector'], v)
                config['models']['student']['kwargs']['reduce'] = v
            elif k == 'p_bswap':
                config['models']['distiller']['compile_kwargs']['background_swap_fn'] = f"eval:BackgroundSwapper(p={v})"
            elif k == 'beta':
                config['models']['distiller']['compile_kwargs']['interpolate_fn'] = f"eval:Interpolater(sampler=torch.distributions.Beta(concentration1={v[0]}, concentration0={v[1]}))"
            elif k == 'lr_X':
                config['models']['synthesizer']['compile_kwargs']['opt_X_kwargs']['kwargs']['lr'] = v
            elif k == 'wandb':
                config['optim']['callbacks']['wandb'] = v

        return expt, config

    elif config['metaconfig']['expt'] == 'vizembed1d':
        expt = expt_vizembed1d
        keys_avail = ['checkpoint', 'num_samples']
        for k, v in custom_args.items():
            assert k in keys_avail, f'Unknown key: {k}'
            if k == 'checkpoint':
                config['optim']['checkpoint'] = bool(v)
            elif k == 'num_samples':
                for d_k in config["datasets"].keys():
                    config["datasets"][d_k]["num_samples"] = v
        return expt, config

    elif config['metaconfig']['expt'] == 'vizembed2d':
        expt = expt_vizembed2d
        keys_avail = ['checkpoint', 'num_samples']
        for k, v in custom_args.items():
            assert k in keys_avail, f'Unknown key: {k}'
            if k == 'checkpoint':
                config['optim']['checkpoint'] = bool(v)
            elif k == 'num_samples':
                for d_k in config["datasets"].keys():
                    config["datasets"][d_k]["num_samples"] = v
        return expt, config

    else:
        raise NotImplementedError(f"Unknown experiment: {config['metaconfig']['expt']}")


def parse_eval_config(config:dict|list|str|Any, parse_flag='eval:'):
    if isinstance(config, dict):
        return {k: parse_eval_config(config=v, parse_flag=parse_flag) for k, v in config.items()}
    elif isinstance(config, list):
        return [parse_eval_config(config=vi, parse_flag=parse_flag) for vi in config]
    elif isinstance(config, str) and config.startswith(parse_flag):
        return eval(config[len(parse_flag):])
    else:
        return config



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddp', type=bool, default=False)
    parser.add_argument('--config', type=str, default='teacher-mnist-lenet5')
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=1)
    # Custom arguments
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--coeffs_T', type=str, default=None)
    parser.add_argument('--beta', type=float, nargs=2, default=None)
    parser.add_argument('--wandb', type=bool, default=None)
    parser = parser.parse_args()
    
    import torch.multiprocessing as mp
    
    custom_args = {
        k: getattr(parser, k) for k in [
            'checkpoint', 'num_epochs', 'num_samples', 'coeffs_T', 'beta',
            'wandb',
        ]
        if getattr(parser, k) is not None
    }

    config = yaml.safe_load(stream=open(file=parse_config_path(config=parser.config, ext='yaml'), mode='r'))
    expt, config = parse_expt(base_config=config, custom_args=custom_args)
    config = parse_eval_config(config, parse_flag='eval:')
    
    for run in range(parser.run, parser.run+parser.n_runs):
        print(f"config: {config['metaconfig']['name']} - run {run}")
        pprint(config)

        if not parser.ddp:
            expt(rank=None, world_size=1, config=config, run=run)
        else:
            world_size = torch.cuda.device_count()
            print(f'world_size: {world_size}')

            setup_master()
            mp.spawn(
                fn=expt,
                args=(world_size, config, run),
                nprocs=world_size,
            )