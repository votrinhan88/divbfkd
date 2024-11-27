# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Any, Callable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE
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

from models.classifiers import ClassifierTrainer, AlexNet, LeNet5, VGG
from models.classifiers.resnetcifar import ResNet_CIFAR
from models.distillers import Distiller, StandardKD, IntermediateFeatureExtractor
from models.GANs.dcgan import Discriminator
from models.GANs.utils import GANGIFMaker
from models.GANs.wgan import WassersteinLoss, WeightClamper
from models.GANs.wgan_gp import WGAN_GP
from utils.callbacks import Callback, ProgressBar, CSVLogger, ModelCheckpoint, SchedulerOnEpoch
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


def expt_teacher(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
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
    model_T:nn.Module = config['models']['teacher']['Class'](**config['models']['teacher']['kwargs'])
    if config['models']['teacher'].get('name') is not None:
        model_T.name = config['models']['teacher']['name']
    trainer = ClassifierTrainer(model=model_T, device=rank, world_size=world_size)
    opt_T = config['optim']['opt_T']['Class'](params=model_T.parameters(), **config['optim']['opt_T']['kwargs'])
    trainer.compile(opt=opt_T)

    # SETUP TRAINING LOOP ######################################################
    callbacks = []
    state_dict_path = f"./logs/{config['dataset']['name']}/{model_T.name} - run {run}.pt"
    if (rank == 0) | (rank is None):
        csv_logger = CSVLogger(
            filename=f"./logs/{config['dataset']['name']}/{model_T.name} - run {run}.csv",
            append=True,
        )
        best_callback = ModelCheckpoint(
            target=model_T,
            filepath=state_dict_path,
            monitor='val_acc',
            save_best_only=True,
        )
        callbacks.extend([csv_logger, best_callback])
    if config['optim'].get('sch_T') is not None:
        sch_T = config['optim']['sch_T']['Class'](optimizer=opt_T, **config['optim']['sch_T']['kwargs'])
        sch_T_cb = SchedulerOnEpoch(on_epoch='train_end', scheduler=sch_T, key='lr')
        callbacks.append(sch_T_cb)
    trainer.training_loop(
        trainloader=dataloader['train'],
        num_epochs=config['optim']['num_epochs'],
        valloader=dataloader['test'],
        callbacks=callbacks,
    )
    model_T.load_state_dict(torch.load(
        f=state_dict_path,
        map_location=trainer.device,
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


def expt_abla_N(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    base_expt = config['main_config']
    print(f'base_expt: {base_expt}')
    expt = parse_expt(base_expt)
    config = CONFIG[base_expt]
    config['dataset']['fewshot_kwargs']['num_samples'] = var
    return expt(rank=rank, world_size=world_size, config=config, run=run)

def expt_abla_M(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    base_expt = config['main_config']

    expt = parse_expt(base_expt)
    config = CONFIG[base_expt]
    config['models']['distiller']['compile_kwargs']['distill_pool_config']['budget'] = var
    return expt(rank=rank, world_size=world_size, config=config, run=run)

def expt_abla_q(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    raise NotImplementedError


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


def viz_embed2d(
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

def viz_embed1d(
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

def expt_viz_embed2d(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    x_embed, y_all, labels = extract_features(rank=rank, world_size=world_size, config=config, run=run, var=var)
    viz_embed2d(
        rank=rank, world_size=world_size, config=config, run=run, var=var,
        x_embed=x_embed, y_all=y_all, labels=labels,
    )
def expt_viz_embed1d(rank:int, world_size:int, config:dict, run:int=0, var:int|float=0):
    x_embed, y_all, labels = extract_features(rank=rank, world_size=world_size, config=config, run=run, var=var)
    viz_embed1d(
        rank=rank, world_size=world_size, config=config, run=run, var=var,
        x_embed=x_embed, y_all=y_all, labels=labels,
    )


def parse_expt(key:str) -> Callable:
    if key[0:4] == 'main':
        return expt_main
    elif key[0:7] == 'teacher':
        return expt_teacher
    elif key[0:11] == 'studentfull':
        return expt_studentfull
    elif key[0:12] == 'studentalone':
        return expt_studentalone
    elif key[0:10] == 'standardkd':
        return expt_standardkd
    elif key[0:4] == 'wgan':
        return expt_wgan
    elif key[0:6] == 'abla-N':
        return expt_abla_N
    elif key[0:6] == 'abla-M':
        return expt_abla_M
    elif key[0:6] == 'abla-q':
        return expt_abla_q
    elif key[0:11] == 'viz-embed2d':
        return expt_viz_embed2d
    elif key[0:11] == 'viz-embed1d':
        return expt_viz_embed1d


CONFIG = {
    'main-mnist-debug': {
        'dataset': {
            'name': 'MNIST',
            'kwargs': {'resize':[32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/MNIST/raw/LeNet5 - 9928.pt',
            },
            'student': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10, 'half_size':True},
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 100, 'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
            },
            'discriminator': {
                'Class': Discriminator,
                'kwargs': {'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'return_logits': True},
            },
            'distiller': {
                'Class': DivBFKD,
                'kwargs': {},
                'compile_kwargs': {
                    'filter': {
                        'mode': 'quantile',     # Filtering method: 'quantile' | 'mean' | 'constant' | None. Defaults to 'quantile' for Adaptive Filtering
                        'per_class': True,      # Flag to set different confidence thresholds for each class. Defaults to True  for Adaptive Filtering
                        'q': 0.1,               # Value of proportion q for mode `quantile`. Defaults to 0.1. (Note: When q=0, random samples can still be introduced into the GAN-training pool)
                        'max_confidence': 0.99, # Upper bound for confidence threshold selection. Defaults to 0.99.
                        'min_confidence': 0.7,  # Lower bound for confidence threshold selection. Defaults to 0.7.
                        # 'mode':'quantile', 'per_class':False, 'q':0.1, 'max_confidence':0.99, 'min_confidence':0.7, # Same thresholds, by quantile
                        # 'mode':'mean', 'per_class':True,                                                            # Mean of confidence per class
                        # 'mode':'mean', 'per_class':False,                                                           # Mean of all confidence
                        # 'mode':'constant', 'value': 0,                                                              # Specified constant(s)
                        # 'mode':None,                                                                                # WGAN
                    },
                    'gan_pool_config': {
                        'budget':0,               # Additional size for combined real + synthetic samples. Defaults to 0 (cut-off to size of few-shot data)
                        'reset':True,             # Introducing synthetic samples to GAN's training set during Adaptive Filtering. Defaults to True.
                        'filter':True,            # Flag to filter the synthetic samples based on confidence thresholds during Adaptive Filtering. Defaults to True.
                        'balanced_classes':False, # Ensure high-confidence images to have balanced classes, unstable with high num_classes. Default: False
                        'aug_gan': get_transform(resize=[32, 32]), # Augmentation function few-shot samples for training GAN. Synthetic samples (via Adaptive Filtering) are not affected. 
                    },
                    'loss_fn_adversarial': True,  # Adversarial loss function for Discriminator and Generator. Defaults to True for WassersteinLoss.
                    'loss_fn_regularize': False,  # Regularization loss function for Discriminator. Defaults to False, (True is CrossEntropyLoss).
                    'loss_fn_distill': True,      # Distillation loss function for training the student in stage 2. Defaults to True for CrossEntropyLoss.
                    'temperature': 1,             # Temperature for softened logits in KD. Defaults to 1 (no softening).
                    'coeff_D_ad': 1,              # Coefficient for adversarial loss. Defaults to 1.
                    'coeff_D_rg': 0,              # Coefficient for regularization loss. Defaults to 0.
                    'coeff_D_gp': 10,             # Coefficient for Gradient Penalty loss following WGAN-GP. Defaults to 10.
                    'repeat_D': 5,                # Ratio of Discriminator:Generator steps. Defaults to 5.
                    'clamp_weights': False,       # Flag to enable weight-clamping to [-0.01, 0.01] (original WGAN).
                    'distill_pool_config': {
                        'budget': 24000,          # Number of additionally generated samples for Distillation set (in stage 2). Defaults to 24000.
                        'filter': False,          # Filter high-confidence samples to Distillation set. Default to False.
                        'balanced_classes':True,  # Flag to enable rejection sampling for balanced classes in Distillation set. Defaults to True.
                        'aug_pseudolabel': get_transform(resize=[32, 32]),      # Augmentation function for collecting pseudolabels from the teacher (stage 1). Should be same to test set's augmentation for stable and accurate pseudolabels.
                        'aug_kd_fewshot': get_transform(resize=[32, 32]),       # Augmentation function for fewshot samples during KD (stage 2). Should be same to train set's augmentation for diversity and consistent knowledge domain.
                        'aug_kd_synth': None,                                   # Augmentation function for synthetic samples during KD (stage 2). These images are already Tensors and have the same dimensions to the student's input so no need for ToTensor() at the end.
                    },
                    'batch_size': 250,            # Batch size for the datapools
                },
            }
        },
        'optim': {
            'num_epochs': [50, 100],
            'opt_G': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr': 5e-5},
            },
            'opt_D': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr': 5e-5},
            },
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-2, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': None,
            'save_G': True,
        }
    },
    'main-mnist': {
        'dataset': {
            'name': 'MNIST',
            'kwargs': {'resize':[32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/MNIST/raw/LeNet5 - 9928.pt',
            },
            'student': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10, 'half_size':True},
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 100, 'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
            },
            'discriminator': {
                'Class': Discriminator,
                'kwargs': {'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'return_logits': True},
            },
            'distiller': {
                'Class': DivBFKD,
                'kwargs': {},
                'compile_kwargs': {
                    'filter': {
                        'mode': 'quantile',     # Filtering method: 'quantile' | 'mean' | 'constant' | None. Defaults to 'quantile' for Adaptive Filtering
                        'per_class': True,      # Flag to set different confidence thresholds for each class. Defaults to True  for Adaptive Filtering
                        'q': 0.1,               # Value of proportion q for mode `quantile`. Defaults to 0.1. (Note: When q=0, random samples can still be introduced into the GAN-training pool)
                        'max_confidence': 0.99, # Upper bound for confidence threshold selection. Defaults to 0.99.
                        'min_confidence': 0.7,  # Lower bound for confidence threshold selection. Defaults to 0.7.
                        # 'mode':'quantile', 'per_class':False, 'q':0.1, 'max_confidence':0.99, 'min_confidence':0.7, # Same thresholds, by quantile
                        # 'mode':'mean', 'per_class':True,                                                            # Mean of confidence per class
                        # 'mode':'mean', 'per_class':False,                                                           # Mean of all confidence
                        # 'mode':'constant', 'value': 0,                                                              # Specified constant(s)
                        # 'mode':None,                                                                                # WGAN
                    },
                    'gan_pool_config': {
                        'budget':0,               # Additional size for combined real + synthetic samples. Defaults to 0 (cut-off to size of few-shot data)
                        'reset':True,             # Introducing synthetic samples to GAN's training set during Adaptive Filtering. Defaults to True.
                        'filter':True,            # Flag to filter the synthetic samples based on confidence thresholds during Adaptive Filtering. Defaults to True.
                        'balanced_classes':False, # Ensure high-confidence images to have balanced classes, unstable with high num_classes. Default: False
                        'aug_gan': get_transform(resize=[32, 32]), # Augmentation function few-shot samples for training GAN. Synthetic samples (via Adaptive Filtering) are not affected. 
                    },
                    'loss_fn_adversarial': True,  # Adversarial loss function for Discriminator and Generator. Defaults to True for WassersteinLoss.
                    'loss_fn_regularize': False,  # Regularization loss function for Discriminator. Defaults to False, (True is CrossEntropyLoss).
                    'loss_fn_distill': True,      # Distillation loss function for training the student in stage 2. Defaults to True for CrossEntropyLoss.
                    'temperature': 1,             # Temperature for softened logits in KD. Defaults to 1 (no softening).
                    'coeff_D_ad': 1,              # Coefficient for adversarial loss. Defaults to 1.
                    'coeff_D_rg': 0,              # Coefficient for regularization loss. Defaults to 0.
                    'coeff_D_gp': 10,             # Coefficient for Gradient Penalty loss following WGAN-GP. Defaults to 10.
                    'repeat_D': 5,                # Ratio of Discriminator:Generator steps. Defaults to 5.
                    'clamp_weights': False,       # Flag to enable weight-clamping to [-0.01, 0.01] (original WGAN).
                    'distill_pool_config': {
                        'budget': 24000,          # Number of additionally generated samples for Distillation set (in stage 2). Defaults to 24000.
                        'filter': False,          # Filter high-confidence samples to Distillation set. Default to False.
                        'balanced_classes':True,  # Flag to enable rejection sampling for balanced classes in Distillation set. Defaults to True.
                        'aug_pseudolabel': get_transform(resize=[32, 32]),      # Augmentation function for collecting pseudolabels from the teacher (stage 1). Should be same to test set's augmentation for stable and accurate pseudolabels.
                        'aug_kd_fewshot': get_transform(resize=[32, 32]),       # Augmentation function for fewshot samples during KD (stage 2). Should be same to train set's augmentation for diversity and consistent knowledge domain.
                        'aug_kd_synth': None,                                   # Augmentation function for synthetic samples during KD (stage 2). These images are already Tensors and have the same dimensions to the student's input so no need for ToTensor() at the end.
                    },
                    'batch_size': 250,            # Batch size for the datapools
                },
            }
        },
        'optim': {
            'num_epochs': [500, 100],
            'opt_G': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr': 5e-5},
            },
            'opt_D': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr': 5e-5},
            },
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-2, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': None,
            'save_G': True,
        }
    },
    'main-fmnist': {
        'dataset': {
            'name': 'FashionMNIST',
            'kwargs': {'resize':[32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {},
                'pretrained_path': './pretrained/FMNIST/raw/LeNet5 - 9090.pt',
            },
            'student': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10, 'half_size':True},
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 200, 'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
            },
            'discriminator': {
                'Class': Discriminator,
                'kwargs': {'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'return_logits': True},
            },
            'distiller': {
                'Class': DivBFKD,
                'kwargs': {},
                'compile_kwargs': {
                    'filter': {
                        'mode': 'quantile',     # Filtering method: 'quantile' | 'mean' | 'constant' | None. Defaults to 'quantile' for Adaptive Filtering
                        'per_class': True,      # Flag to set different confidence thresholds for each class. Defaults to True  for Adaptive Filtering
                        'q': 0.1,               # Value of proportion q for mode `quantile`. Defaults to 0.1. (Note: When q=0, random samples can still be introduced into the GAN-training pool)
                        'max_confidence': 0.99, # Upper bound for confidence threshold selection. Defaults to 0.99.
                        'min_confidence': 0.7,  # Lower bound for confidence threshold selection. Defaults to 0.7.
                        # 'mode':'quantile', 'per_class':False, 'q':0.1, 'max_confidence':0.99, 'min_confidence':0.7, # Same thresholds, by quantile
                        # 'mode':'mean', 'per_class':True,                                                            # Mean of confidence per class
                        # 'mode':'mean', 'per_class':False,                                                           # Mean of all confidence
                        # 'mode':'constant', 'value': 0,                                                              # Specified constant(s)
                        # 'mode':None,                                                                                # WGAN
                    },
                    'gan_pool_config': {
                        'budget':0,               # Additional size for combined real + synthetic samples. Defaults to 0 (cut-off to size of few-shot data)
                        'reset':True,             # Introducing synthetic samples to GAN's training set during Adaptive Filtering. Defaults to True.
                        'filter':True,            # Flag to filter the synthetic samples based on confidence thresholds during Adaptive Filtering. Defaults to True.
                        'balanced_classes':False, # Ensure high-confidence images to have balanced classes, unstable with high num_classes. Default: False
                        'aug_gan': get_transform(resize=[32, 32]), # Augmentation function few-shot samples for training GAN. Synthetic samples (via Adaptive Filtering) are not affected. 
                    },
                    'loss_fn_adversarial': True,  # Adversarial loss function for Discriminator and Generator. Defaults to True for WassersteinLoss.
                    'loss_fn_regularize': False,  # Regularization loss function for Discriminator. Defaults to False, (True is CrossEntropyLoss).
                    'loss_fn_distill': True,      # Distillation loss function for training the student in stage 2. Defaults to True for CrossEntropyLoss.
                    'temperature': 1,             # Temperature for softened logits in KD. Defaults to 1 (no softening).
                    'coeff_D_ad': 1,              # Coefficient for adversarial loss. Defaults to 1.
                    'coeff_D_rg': 0,              # Coefficient for regularization loss. Defaults to 0.
                    'coeff_D_gp': 10,             # Coefficient for Gradient Penalty loss following WGAN-GP. Defaults to 10.
                    'repeat_D': 5,                # Ratio of Discriminator:Generator steps. Defaults to 5.
                    'clamp_weights': False,       # Flag to enable weight-clamping to [-0.01, 0.01] (original WGAN).
                    'distill_pool_config': {
                        'budget': 48000,          # Number of additionally generated samples for Distillation set (in stage 2). Defaults to 24000.
                        'filter': False,          # Filter high-confidence samples to Distillation set. Default to False.
                        'balanced_classes':True,  # Flag to enable rejection sampling for balanced classes in Distillation set. Defaults to True.
                        'aug_pseudolabel': get_transform(resize=[32, 32]),      # Augmentation function for collecting pseudolabels from the teacher (stage 1). Should be same to test set's augmentation for stable and accurate pseudolabels.
                        'aug_kd_fewshot': get_transform(resize=[32, 32]),       # Augmentation function for fewshot samples during KD (stage 2). Should be same to train set's augmentation for diversity and consistent knowledge domain.
                        'aug_kd_synth': None,                                   # Augmentation function for synthetic samples during KD (stage 2). These images are already Tensors and have the same dimensions to the student's input so no need for ToTensor() at the end.
                    },
                    'aug_S': None,                # Augmentation for training the student in Stage 2. Defaults to None.
                    'batch_size': 250,            # Batch size for the datapools
                },
            }
        },
        'optim': {
            'num_epochs': [1000, 200],
            'opt_G': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr': 5e-5},
            },
            'opt_D': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr': 5e-5},
            },
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-2, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': None,
            'save_G': True,
        }
    },
    'main-svhn': {
        'dataset': {
            'name': 'SVHN',
            'kwargs': {
                'splits': ['train', 'test'],
                'init_augmentations': {'train': transforms.RandomCrop(32, padding=4)},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': AlexNet,
                'kwargs': {'input_dim': [3, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/SVHN/raw/AlexNet - 9616.pt',
            },
            'student': {
                'Class': AlexNet,
                'kwargs': {'half_size': True, 'input_dim': [3, 32, 32], 'num_classes': 10},
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 256, 'image_dim': [3, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
            },
            'discriminator': {
                'Class': Discriminator,
                'kwargs': {'image_dim': [3, 32, 32], 'base_dim': [256, 8, 8], 'return_logits': True},
            },
            'distiller': {
                'Class': DivBFKD,
                'kwargs': {},
                'compile_kwargs': {
                    'filter': {
                        'mode': 'quantile',     # Filtering method: 'quantile' | 'mean' | 'constant' | None. Defaults to 'quantile' for Adaptive Filtering
                        'per_class': True,      # Flag to set different confidence thresholds for each class. Defaults to True  for Adaptive Filtering
                        'q': 0.1,               # Value of proportion q for mode `quantile`. Defaults to 0.1. (Note: When q=0, random samples can still be introduced into the GAN-training pool)
                        'max_confidence': 0.99, # Upper bound for confidence threshold selection. Defaults to 0.99.
                        'min_confidence': 0.7,  # Lower bound for confidence threshold selection. Defaults to 0.7.
                    },
                    'gan_pool_config': {
                        'budget':0,               # Additional size for combined real + synthetic samples. Defaults to 0 (cut-off to size of few-shot data)
                        'reset':True,             # Introducing synthetic samples to GAN's training set during Adaptive Filtering. Defaults to True.
                        'filter':True,            # Flag to filter the synthetic samples based on confidence thresholds during Adaptive Filtering. Defaults to True.
                        'balanced_classes':False, # Ensure high-confidence images to have balanced classes, unstable with high num_classes. Default: False
                        'aug_gan': transforms.ToTensor(), # Augmentation function few-shot samples for training GAN. Synthetic samples (via Adaptive Filtering) are not affected. 
                    },
                    'loss_fn_adversarial': True,  # Adversarial loss function for Discriminator and Generator. Defaults to True for WassersteinLoss.
                    'loss_fn_regularize': False,  # Regularization loss function for Discriminator. Defaults to False, (True is CrossEntropyLoss).
                    'loss_fn_distill': True,      # Distillation loss function for training the student in stage 2. Defaults to True for CrossEntropyLoss.
                    'temperature': 1,             # Temperature for softened logits in KD. Defaults to 1 (no softening).
                    'coeff_D_ad': 1,              # Coefficient for adversarial loss. Defaults to 1.
                    'coeff_D_rg': 0,              # Coefficient for regularization loss. Defaults to 0.
                    'coeff_D_gp': 10,             # Coefficient for Gradient Penalty loss following WGAN-GP. Defaults to 10.
                    'repeat_D': 5,                # Ratio of Discriminator:Generator steps. Defaults to 5.
                    'clamp_weights': False,       # Flag to enable weight-clamping to [-0.01, 0.01] (original WGAN).
                    'distill_pool_config': {
                        'budget': 40000,          # Number of additionally generated samples for Distillation set (in stage 2). Defaults to 24000.
                        'filter': False,          # Filter high-confidence samples to Distillation set. Default to False.
                        'balanced_classes':True,  # Flag to enable rejection sampling for balanced classes in Distillation set. Defaults to True.
                        'aug_pseudolabel': transforms.ToTensor(),               # Augmentation function for collecting pseudolabels from the teacher (stage 1). Should be same to test set's augmentation for stable and accurate pseudolabels.
                        'aug_kd_fewshot': transforms.Compose([                  # Augmentation function for fewshot samples during KD (stage 2). Should be same to train set's augmentation for diversity and consistent knowledge domain.
                            transforms.RandomCrop(32, padding=4),
                            transforms.ToTensor(),
                        ]),                                                     
                        'aug_kd_synth': transforms.RandomCrop(32, padding=4),   # Augmentation function for synthetic samples during KD (stage 2). These images are already Tensors and have the same dimensions to the student's input so no need for ToTensor() at the end.
                    },
                    'batch_size': 250,            # Batch size for the datapools
                },
            }
        },
        'optim': {
            'num_epochs': [1000, 200],
            'opt_G': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr': 5e-5},
            },
            'opt_D': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr': 5e-5},
            },
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones': [1100, 1150], 'gamma':0.1},
            },
            'save_G': True,
        },
    },
    'main-cifar10': {
        'dataset': {
            'name': 'CIFAR10',
            'kwargs': {
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(size=[32, 32], padding=4),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': AlexNet,
                'kwargs': {'input_dim': [3, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/CIFAR10/raw/AlexNet - 9005.pt',
            },
            'student': {
                'Class': AlexNet,
                'kwargs': {'half_size': True, 'input_dim': [3, 32, 32], 'num_classes': 10},
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 256, 'image_dim': [3, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
            },
            'discriminator': {
                'Class': Discriminator,
                'kwargs': {'image_dim': [3, 32, 32], 'base_dim': [256, 8, 8], 'return_logits': True},
            },
            'distiller': {
                'Class': DivBFKD,
                'kwargs': {},
                'compile_kwargs': {
                    'filter': {
                        'mode': 'quantile',     # Filtering method: 'quantile' | 'mean' | 'constant' | None. Defaults to 'quantile' for Adaptive Filtering
                        'per_class': True,      # Flag to set different confidence thresholds for each class. Defaults to True  for Adaptive Filtering
                        'q': 0.1,               # Value of proportion q for mode `quantile`. Defaults to 0.1. (Note: When q=0, random samples can still be introduced into the GAN-training pool)
                        'max_confidence': 0.99, # Upper bound for confidence threshold selection. Defaults to 0.99.
                        'min_confidence': 0.7,  # Lower bound for confidence threshold selection. Defaults to 0.7.
                    },
                    'gan_pool_config': {
                        'budget':0,               # Additional size for combined real + synthetic samples. Defaults to 0 (cut-off to size of few-shot data)
                        'reset':True,             # Introducing synthetic samples to GAN's training set during Adaptive Filtering. Defaults to True.
                        'filter':True,            # Flag to filter the synthetic samples based on confidence thresholds during Adaptive Filtering. Defaults to True.
                        'balanced_classes':False, # Ensure high-confidence images to have balanced classes, unstable with high num_classes. Default: False
                        'aug_gan': transforms.ToTensor(), # Augmentation function few-shot samples for training GAN. Synthetic samples (via Adaptive Filtering) are not affected. 
                    },
                    'loss_fn_adversarial': True,  # Adversarial loss function for Discriminator and Generator. Defaults to True for WassersteinLoss.
                    'loss_fn_regularize': False,  # Regularization loss function for Discriminator. Defaults to False, (True is CrossEntropyLoss).
                    'loss_fn_distill': True,      # Distillation loss function for training the student in stage 2. Defaults to True for CrossEntropyLoss.
                    'temperature': 1,             # Temperature for softened logits in KD. Defaults to 1 (no softening).
                    'coeff_D_ad': 1,              # Coefficient for adversarial loss. Defaults to 1.
                    'coeff_D_rg': 0,              # Coefficient for regularization loss. Defaults to 0.
                    'coeff_D_gp': 10,             # Coefficient for Gradient Penalty loss following WGAN-GP. Defaults to 10.
                    'repeat_D': 5,                # Ratio of Discriminator:Generator steps. Defaults to 5.
                    'clamp_weights': False,       # Flag to enable weight-clamping to [-0.01, 0.01] (original WGAN).
                    'distill_pool_config': {
                        'budget': 40000,          # Number of additionally generated samples for Distillation set (in stage 2). Defaults to 24000.
                        'filter': False,          # Filter high-confidence samples to Distillation set. Default to False.
                        'balanced_classes':True,  # Flag to enable rejection sampling for balanced classes in Distillation set. Defaults to True.
                        'aug_pseudolabel': transforms.ToTensor(),               # Augmentation function for collecting pseudolabels from the teacher (stage 1). Should be same to test set's augmentation for stable and accurate pseudolabels.
                        'aug_kd_fewshot': transforms.Compose([                  # Augmentation function for fewshot samples during KD (stage 2). Should be same to train set's augmentation for diversity and consistent knowledge domain.
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),                                                     
                        'aug_kd_synth': transforms.Compose([                    # Augmentation function for synthetic samples during KD (stage 2). These images are already Tensors and have the same dimensions to the student's input so no need for ToTensor() at the end.
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                        ]),                                                     
                    },
                    'batch_size': 250,            # Batch size for the datapools
                },
            }
        },
        'optim': {
            'num_epochs': [2000, 400],
            'opt_G': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr': 5e-5},
            },
            'opt_D': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr': 5e-5},
            },
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones': [2200, 2300], 'gamma':0.1},
            },
            'save_G': True,
        },
    },
    'main-cifar100': {
        'dataset': {
            'name': 'CIFAR100',
            'kwargs': {
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(size=[32, 32], padding=4),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 100,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 5000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': ResNet_CIFAR,
                'kwargs': {'depth':32, 'skip_option':'A', 'input_dim': [3, 32, 32], 'num_classes': 100},
                'pretrained_path': './pretrained/CIFAR100/raw/ResNet32_CIFAR - 7141.pt',
            },
            'student': {
                'Class': ResNet_CIFAR,
                'kwargs': {'depth':20, 'skip_option':'A', 'input_dim': [3, 32, 32], 'num_classes': 100},
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 256, 'image_dim': [3, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
            },
            'discriminator': {
                'Class': Discriminator,
                'kwargs': {'image_dim': [3, 32, 32], 'base_dim': [256, 8, 8], 'return_logits': True},
            },
            'distiller': {
                'Class': DivBFKD,
                'kwargs': {},
                'compile_kwargs': {
                    'filter': {
                        'mode': 'quantile',     # Filtering method: 'quantile' | 'mean' | 'constant' | None. Defaults to 'quantile' for Adaptive Filtering
                        'per_class': True,      # Flag to set different confidence thresholds for each class. Defaults to True  for Adaptive Filtering
                        'q': 0.1,               # Value of proportion q for mode `quantile`. Defaults to 0.1. (Note: When q=0, random samples can still be introduced into the GAN-training pool)
                        'max_confidence': 0.99, # Upper bound for confidence threshold selection. Defaults to 0.99.
                        'min_confidence': 0.7,  # Lower bound for confidence threshold selection. Defaults to 0.7.
                    },
                    'gan_pool_config': {
                        'budget':0,               # Additional size for combined real + synthetic samples. Defaults to 0 (cut-off to size of few-shot data)
                        'reset':True,             # Introducing synthetic samples to GAN's training set during Adaptive Filtering. Defaults to True.
                        'filter':True,            # Flag to filter the synthetic samples based on confidence thresholds during Adaptive Filtering. Defaults to True.
                        'balanced_classes':False, # Ensure high-confidence images to have balanced classes, unstable with high num_classes. Default: False
                        'aug_gan': transforms.ToTensor(), # Augmentation function few-shot samples for training GAN. Synthetic samples (via Adaptive Filtering) are not affected. 
                    },
                    'loss_fn_adversarial': True,  # Adversarial loss function for Discriminator and Generator. Defaults to True for WassersteinLoss.
                    'loss_fn_regularize': False,  # Regularization loss function for Discriminator. Defaults to False, (True is CrossEntropyLoss).
                    'loss_fn_distill': True,      # Distillation loss function for training the student in stage 2. Defaults to True for CrossEntropyLoss.
                    'temperature': 1,             # Temperature for softened logits in KD. Defaults to 1 (no softening).
                    'coeff_D_ad': 1,              # Coefficient for adversarial loss. Defaults to 1.
                    'coeff_D_rg': 0,              # Coefficient for regularization loss. Defaults to 0.
                    'coeff_D_gp': 10,             # Coefficient for Gradient Penalty loss following WGAN-GP. Defaults to 10.
                    'repeat_D': 5,                # Ratio of Discriminator:Generator steps. Defaults to 5.
                    'clamp_weights': False,       # Flag to enable weight-clamping to [-0.01, 0.01] (original WGAN).
                    'distill_pool_config': {
                        'budget': 40000,          # Number of additionally generated samples for Distillation set (in stage 2). Defaults to 24000.
                        'filter': False,          # Filter high-confidence samples to Distillation set. Default to False.
                        'balanced_classes':True,  # Flag to enable rejection sampling for balanced classes in Distillation set. Defaults to True.
                        'aug_pseudolabel': transforms.ToTensor(),               # Augmentation function for collecting pseudolabels from the teacher (stage 1). Should be same to test set's augmentation for stable and accurate pseudolabels.
                        'aug_kd_fewshot': transforms.Compose([                  # Augmentation function for fewshot samples during KD (stage 2). Should be same to train set's augmentation for diversity and consistent knowledge domain.
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),                                                     
                        'aug_kd_synth': transforms.Compose([                    # Augmentation function for synthetic samples during KD (stage 2). These images are already Tensors and have the same dimensions to the student's input so no need for ToTensor() at the end.
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                        ]),                                                     
                    },
                    'batch_size': 250,            # Batch size for the datapools
                },
            }
        },
        'optim': {
            'num_epochs': [5000, 1000],
            'opt_G': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr': 5e-5},
            },
            'opt_D': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr': 5e-5},
            },
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones': [5500, 5750], 'gamma':0.1},
            },
            'save_G': True,
        },
    },
    'main-tinyimagenet': {
        'dataset': {
            'name': 'TinyImageNet',
            'kwargs': {
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(size=[64, 64], padding=8),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 64, 64],
            'num_classes': 200,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 10000, 'balanced': True, 'shuffle':True},
            'split_remap': {'val': 'test'},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': ResNet_CIFAR,
                'kwargs': {'depth':32, 'skip_option':'A', 'input_dim': [3, 64, 64], 'num_classes': 200},
                'pretrained_path': './pretrained/TinyImageNet/raw/ResNet32_CIFAR - 5627.pt',
            },
            'student': {'Class': ResNet_CIFAR, 'kwargs': {'depth':20, 'skip_option':'A', 'input_dim': [3, 64, 64], 'num_classes': 200}},
            'generator': {'Class': Generator, 'kwargs': {'latent_dim': 256, 'image_dim': [3, 64, 64], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d}},
            'discriminator': {'Class': Discriminator,'kwargs': {'image_dim': [3, 64, 64], 'base_dim': [256, 8, 8], 'return_logits': True}},
            'distiller': {
                'Class': DivBFKD,
                'kwargs': {},
                'compile_kwargs': {
                    'filter': {
                        'mode': 'quantile',     # Filtering method: 'quantile' | 'mean' | 'constant' | None. Defaults to 'quantile' for Adaptive Filtering
                        'per_class': True,      # Flag to set different confidence thresholds for each class. Defaults to True  for Adaptive Filtering
                        'q': 0.1,               # Value of proportion q for mode `quantile`. Defaults to 0.1. (Note: When q=0, random samples can still be introduced into the GAN-training pool)
                        'max_confidence': 0.99, # Upper bound for confidence threshold selection. Defaults to 0.99.
                        'min_confidence': 0.7,  # Lower bound for confidence threshold selection. Defaults to 0.7.
                    },
                    'gan_pool_config': {
                        'budget':0,               # Additional size for combined real + synthetic samples. Defaults to 0 (cut-off to size of few-shot data)
                        'reset':True,             # Introducing synthetic samples to GAN's training set during Adaptive Filtering. Defaults to True.
                        'filter':True,            # Flag to filter the synthetic samples based on confidence thresholds during Adaptive Filtering. Defaults to True.
                        'balanced_classes':False, # Ensure high-confidence images to have balanced classes, unstable with high num_classes. Default: False
                        'aug_gan': transforms.ToTensor(), # Augmentation function few-shot samples for training GAN. Synthetic samples (via Adaptive Filtering) are not affected. 
                    },
                    'loss_fn_adversarial': True,  # Adversarial loss function for Discriminator and Generator. Defaults to True for WassersteinLoss.
                    'loss_fn_regularize': False,  # Regularization loss function for Discriminator. Defaults to False, (True is CrossEntropyLoss).
                    'loss_fn_distill': True,      # Distillation loss function for training the student in stage 2. Defaults to True for CrossEntropyLoss.
                    'temperature': 1,             # Temperature for softened logits in KD. Defaults to 1 (no softening).
                    'coeff_D_ad': 1,              # Coefficient for adversarial loss. Defaults to 1.
                    'coeff_D_rg': 0,              # Coefficient for regularization loss. Defaults to 0.
                    'coeff_D_gp': 10,             # Coefficient for Gradient Penalty loss following WGAN-GP. Defaults to 10.
                    'repeat_D': 5,                # Ratio of Discriminator:Generator steps. Defaults to 5.
                    'clamp_weights': False,       # Flag to enable weight-clamping to [-0.01, 0.01] (original WGAN).
                    'distill_pool_config': {
                        'budget': 50000,          # Number of additionally generated samples for Distillation set (in stage 2). Defaults to 24000.
                        'filter': False,          # Filter high-confidence samples to Distillation set. Default to False.
                        'balanced_classes':True,  # Flag to enable rejection sampling for balanced classes in Distillation set. Defaults to True.
                        'aug_pseudolabel': transforms.ToTensor(),               # Augmentation function for collecting pseudolabels from the teacher (stage 1). Should be same to test set's augmentation for stable and accurate pseudolabels.
                        'aug_kd_fewshot': transforms.Compose([                  # Augmentation function for fewshot samples during KD (stage 2). Should be same to train set's augmentation for diversity and consistent knowledge domain.
                            transforms.RandomCrop(64, padding=8),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),                                                     
                        'aug_kd_synth': transforms.Compose([                    # Augmentation function for synthetic samples during KD (stage 2). These images are already Tensors and have the same dimensions to the student's input so no need for ToTensor() at the end.
                            transforms.RandomCrop(64, padding=8),
                            transforms.RandomHorizontalFlip(),
                        ]),                                                     
                    },
                    'batch_size': 250,            # Batch size for the datapools
                },
            }
        },
        'optim': {
            'num_epochs': [5000, 1000],
            'opt_G': {'Class': torch.optim.RMSprop, 'kwargs': {'lr': 5e-5}},
            'opt_D': {'Class': torch.optim.RMSprop, 'kwargs': {'lr': 5e-5}},
            'opt_S': {'Class': torch.optim.SGD, 'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4}},
            'sch_S': {'Class': torch.optim.lr_scheduler.MultiStepLR, 'kwargs': {'milestones': [5500, 5750], 'gamma':0.1}},
            'save_G': True,
        },
    },
    'main-imagenette': {
        'dataset': {
            'name': 'Imagenette',
            'kwargs': {
                'init_augmentations': {
                    'train': transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                    ]),
                    'val': transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                    ]),
                },
            },
            'image_dim': [3, 224, 224],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'split_remap': {'val': 'test'},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': torchvision.models.resnet34,
                'kwargs': {'num_classes': 10},
                'pretrained_path': './pretrained/Imagenette/raw/ResNet34 - 9047.pt',
                'name': 'ResNet34',
            },
            'student': {
                'Class': torchvision.models.resnet18,
                'kwargs': {'num_classes': 10},
                'name': 'ResNet18',
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 256, 'image_dim': [3, 224, 224], 'base_dim': [256, 7, 7], 'NormLayer': nn.BatchNorm2d},
            },
            'discriminator': {
                'Class': Discriminator,
                'kwargs': {'image_dim': [3, 224, 224], 'base_dim': [256, 7, 7], 'return_logits': True},
            },
            'distiller': {
                'Class': DivBFKD,
                'kwargs': {'image_dim': [3, 224, 224], 'num_classes': 10},
                'compile_kwargs': {
                    'filter': {
                        'mode': 'quantile',     # Filtering method: 'quantile' | 'mean' | 'constant' | None. Defaults to 'quantile' for Adaptive Filtering
                        'per_class': True,      # Flag to set different confidence thresholds for each class. Defaults to True  for Adaptive Filtering
                        'q': 0.1,               # Value of proportion q for mode `quantile`. Defaults to 0.1. (Note: When q=0, random samples can still be introduced into the GAN-training pool)
                        'max_confidence': 0.99, # Upper bound for confidence threshold selection. Defaults to 0.99.
                        'min_confidence': 0.7,  # Lower bound for confidence threshold selection. Defaults to 0.7.
                    },
                    'gan_pool_config': {
                        'budget':0,               # Additional size for combined real + synthetic samples. Defaults to 0 (cut-off to size of few-shot data)
                        'reset':True,             # Introducing synthetic samples to GAN's training set during Adaptive Filtering. Defaults to True.
                        'filter':True,            # Flag to filter the synthetic samples based on confidence thresholds during Adaptive Filtering. Defaults to True.
                        'balanced_classes':False, # Ensure high-confidence images to have balanced classes, unstable with high num_classes. Default: False
                        'aug_gan': transforms.Compose([     # Augmentation function few-shot samples for training GAN. Synthetic samples (via Adaptive Filtering) are not affected. 
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                        ]),
                    },
                    'loss_fn_adversarial': True,  # Adversarial loss function for Discriminator and Generator. Defaults to True for WassersteinLoss.
                    'loss_fn_regularize': False,  # Regularization loss function for Discriminator. Defaults to False, (True is CrossEntropyLoss).
                    'loss_fn_distill': True,      # Distillation loss function for training the student in stage 2. Defaults to True for CrossEntropyLoss.
                    'temperature': 1,             # Temperature for softened logits in KD. Defaults to 1 (no softening).
                    'coeff_D_ad': 1,              # Coefficient for adversarial loss. Defaults to 1.
                    'coeff_D_rg': 0,              # Coefficient for regularization loss. Defaults to 0.
                    'coeff_D_gp': 10,             # Coefficient for Gradient Penalty loss following WGAN-GP. Defaults to 10.
                    'repeat_D': 5,                # Ratio of Discriminator:Generator steps. Defaults to 5.
                    'clamp_weights': False,       # Flag to enable weight-clamping to [-0.01, 0.01] (original WGAN).
                    'distill_pool_config': {
                        'budget': 40000,          # Number of additionally generated samples for Distillation set (in stage 2). Defaults to 24000.
                        'filter': False,          # Filter high-confidence samples to Distillation set. Default to False.
                        'balanced_classes':True,  # Flag to enable rejection sampling for balanced classes in Distillation set. Defaults to True.
                        'aug_pseudolabel': transforms.Compose([                 # Augmentation function for collecting pseudolabels from the teacher (stage 1). Should be same to test set's augmentation for stable and accurate pseudolabels.
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                        ]),
                        'aug_kd_fewshot': transforms.Compose([                  # Augmentation function for fewshot samples during KD (stage 2). Should be same to train set's augmentation for diversity and consistent knowledge domain.
                            transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),                                                     
                        'aug_kd_synth': transforms.Compose([                    # Augmentation function for synthetic samples during KD (stage 2). These images are already Tensors and have the same dimensions to the student's input so no need for ToTensor() at the end.
                            transforms.RandomCrop(224, padding=32), # 28 is to-scale but 32 makes more sense (sum = 256)
                            transforms.RandomHorizontalFlip(),
                        ]),                                                     
                    },
                    'batch_size': 250,            # Batch size for the datapools
                },
            }
        },
        'optim': {
            'num_epochs': [2000, 400],
            'opt_G': {'Class': torch.optim.RMSprop, 'kwargs': {'lr': 5e-5}},
            'opt_D': {'Class': torch.optim.RMSprop, 'kwargs': {'lr': 5e-5}},
            'opt_S': {'Class': torch.optim.SGD, 'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4}},
            'sch_S': {'Class': torch.optim.lr_scheduler.MultiStepLR, 'kwargs': {'milestones': [2200, 2300], 'gamma':0.1}},
            'save_G': True,
        },
    },
    
    'teacher-mnist': {
        'dataset': {
            'name': 'MNIST',
            'kwargs': {'resize':[32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'num_workers': 4,
        },
        'models': {'teacher': {
            'Class': LeNet5,
            'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
        }},
        'optim': {
            'num_epochs': 100,
            'opt_T': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-2, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_T': None,
        },
    },
    'teacher-fmnist': {
        'dataset': {
            'name': 'FashionMNIST',
            'kwargs': {'resize':[32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'num_workers': 4,
        },
        'models': {'teacher': {
            'Class': LeNet5,
            'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
        }},
        'optim': {
            'num_epochs': 100,
            'opt_T': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-2, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_T': None,
        },
    },
    'teacher-svhn': {
        'dataset': {
            'name': 'SVHN',
            'kwargs': {
                'splits': ['train', 'test'],
                'init_augmentations': {'train': transforms.RandomCrop(32, padding=4)},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'num_workers': 4,
        },
        'models': {'teacher': {
            'Class': AlexNet,
            'kwargs': {'input_dim': [3, 32, 32], 'num_classes': 10},
        }},
        'optim': {
            'num_epochs': 200,
            'opt_T': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_T': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones':[100, 150], 'gamma':0.1},
            },
        },
    },
    'teacher-cifar10': {
        'dataset': {
            'name': 'CIFAR10',
            'kwargs': {
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'num_workers': 4,
        },
        'models': {'teacher': {
            'Class': AlexNet,
            'kwargs': {'input_dim': [3, 32, 32], 'num_classes': 10},
        }},
        'optim': {
            'num_epochs': 200,
            'opt_T': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_T': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones':[100, 150], 'gamma':0.1},
            },
        },
    },
    'teacher-cifar100': {
        'dataset': {
            'name': 'CIFAR100',
            'kwargs': {
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 100,
            'onehot_label': False,
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'num_workers': 4,
        },
        'models': { 'teacher': {
            'Class': ResNet_CIFAR,
            'kwargs': {
                'input_dim': [3, 32, 32],
                'num_classes': 100,
                'depth':32,
                'skip_option':'A',
            }
        }},
        'optim': {
            'num_epochs': 200,
            'opt_T': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_T': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones':[100, 150], 'gamma':0.1},
            },
        },
    },
    'teacher-tinyimagenet': {
        'dataset': {
            'name': 'TinyImageNet',
            'kwargs': {
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(size=[64, 64], padding=8),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 64, 64],
            'num_classes': 200,
            'onehot_label': False,
            'split_remap': {'val':'test'},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'val':False}},
            'batch_size': {'train': 128, 'val': 1024},
            'num_workers': 4,
        },
        'models': {'teacher': {
            'Class': ResNet_CIFAR,
            'kwargs': {
                'input_dim': [3, 64, 64],
                'num_classes': 200,
                'depth':32,
                'skip_option':'A',
            }},
        },
        'optim': {
            'num_epochs': 200,
            'opt_T': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_T': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones':[100, 150], 'gamma':0.1},
            },
        },
    },
    'teacher-imagenette': {
        'dataset': {
            'name': 'Imagenette',
            'kwargs': {
                'init_augmentations': {
                    'train': transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                    ]),
                    'val': transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                    ]),
                },
            },
            'image_dim': [3, 224, 224],
            'num_classes': 10,
            'onehot_label': False,
            'split_remap': {'val': 'test'},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 128},
            'kwargs': {'num_workers': 4, 'prefetch_factor': 2, 'pin_memory': True}
        },
        'models': {'teacher': {
            'Class': torchvision.models.resnet34,
            'kwargs': {'num_classes': 10},
            'name': 'ResNet34',
        }},
        'optim': {
            'num_epochs': 200,
            'opt_T': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_T': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones':[100, 150], 'gamma':0.1},
            },
        },
    },

    'studentfull-mnist': {
        'dataset': {
            'name': 'MNIST',
            'kwargs': {'resize':[32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs':{'num_workers': 4,},
        },
        'models': {'student': {
            'Class': LeNet5,
            'kwargs': {'half_size':True, 'input_dim': [1, 32, 32], 'num_classes': 10},
        }},
        'optim': {
            'num_epochs': 100,
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-2, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': None,
        },
    },
    'studentfull-fmnist': {
        'dataset': {
            'name': 'FashionMNIST',
            'kwargs': {'resize':[32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs':{'num_workers': 4,},
        },
        'models': {'student': {
            'Class': LeNet5,
            'kwargs': {'half_size':True, 'input_dim': [1, 32, 32], 'num_classes': 10},
        }},
        'optim': {
            'num_epochs': 100,
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-2, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': None,
        },
    },
    'studentfull-svhn': {NotImplementedError},
    'studentfull-cifar10': {NotImplementedError},
    'studentfull-cifar100': {NotImplementedError},
    'studentfull-tinyimagenet': {NotImplementedError},
    'studentfull-imagenette': {
        'dataset': {
            'name': 'Imagenette',
            'kwargs': {
                'init_augmentations': {
                    'train': transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                    ]),
                    'val': transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                    ]),
                },
            },
            'image_dim': [3, 224, 224],
            'num_classes': 10,
            'onehot_label': False,
            'split_remap': {'val': 'test'},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 128},
            'kwargs': {'num_workers': 4, 'prefetch_factor': 2, 'pin_memory': True}
        },
        'models': {'student': {
            'Class': torchvision.models.resnet18,
            'kwargs': {'num_classes': 10},
            'name': 'ResNet18',
        }},
        'optim': {
            'num_epochs': 200,
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones':[100, 150], 'gamma':0.1},
            },
        },
    },
    
    'studentalone-mnist': {
        'dataset': {
            'name': 'MNIST',
            'kwargs': {'resize':[32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {'student': {
            'Class': LeNet5,
            'kwargs': {'half_size':True, 'input_dim': [1, 32, 32], 'num_classes': 10},
        }},
        'optim': {
            'num_epochs': 100,
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-2, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': None,
        },
    },
    'studentalone-fmnist': {NotImplementedError},
    'studentalone-svhn': {NotImplementedError},
    'studentalone-cifar10': {NotImplementedError},
    'studentalone-cifar100': {NotImplementedError},
    'studentalone-tinyimagenet': {NotImplementedError},
    'studentalone-imagenette': {
        'dataset': {
            'name': 'Imagenette',
            'kwargs': {
                'init_augmentations': {
                    'train': transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                    ]),
                    'val': transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                    ]),
                },
            },
            'image_dim': [3, 224, 224],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'split_remap': {'val': 'test'},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {'num_workers': 4, 'prefetch_factor': 2, 'pin_memory': True}
        },
        'models': {'student': {
            'Class': torchvision.models.resnet18,
            'kwargs': {'num_classes': 10},
            'name': 'ResNet18',
        }},
        'optim': {
            'num_epochs': 200,
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones':[100, 150], 'gamma':0.1},
            },
        },
    },

    'standardkd-mnist':{
        'dataset': {
            'name': 'MNIST',
            'kwargs': {'resize':[32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/MNIST/raw/LeNet5 - 9928.pt',
            },
            'student': {
                'Class': LeNet5,
                'kwargs': {'half_size':True, 'input_dim': [1, 32, 32], 'num_classes': 10},
            },
            'distiller': {
                'Class': HintonDistiller,
                'compile_kwargs': {
                    'loss_fn_distill': nn.KLDivLoss(reduction='batchmean'),
                    'coeff_dt': 0.9,
                    'coeff_lb': 0.1,
                    'temperature':1,
                },
            }
        },
        'optim': {
            'num_epochs': 100,
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-2, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': None,
        },
    },
    'standardkd-fmnist':{
        'dataset': {
            'name': 'FashionMNIST',
            'kwargs': {'resize':[32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/FMNIST/raw/LeNet5 - 9090.pt',
            },
            'student': {
                'Class': LeNet5,
                'kwargs': {'half_size':True, 'input_dim': [1, 32, 32], 'num_classes': 10},
            },
            'distiller': {
                'Class': HintonDistiller,
                'compile_kwargs': {
                    'loss_fn_distill': nn.KLDivLoss(reduction='batchmean'),
                    'coeff_dt': 0.9,
                    'coeff_lb': 0.1,
                    'temperature':1,
                },
            }
        },
        'optim': {
            'num_epochs': 100,
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-2, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': None,
        }
    },
    'standardkd-svhn':{
        'dataset': {
            'name': 'SVHN',
            'kwargs': {
                'splits': ['train', 'test'],
                'resize':[32, 32],
                'init_augmentations': {'train': transforms.RandomCrop(32, padding=4)},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': AlexNet,
                'kwargs': {'input_dim': [3, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/SVHN/raw/AlexNet - 9616.pt',
            },
            'student': {
                'Class': AlexNet,
                'kwargs': {'half_size': True, 'input_dim': [3, 32, 32], 'num_classes': 10},
            },
            'distiller': {
                'Class': HintonDistiller,
                'compile_kwargs': {
                    'loss_fn_distill': nn.KLDivLoss(reduction='batchmean'),
                    'coeff_dt': 0.9,
                    'coeff_lb': 0.1,
                    'temperature':1,
                },
            }
        },
        'optim': {
            'num_epochs': 200,
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones':[100, 150], 'gamma':0.1},
            },
        }
    },
    'standardkd-cifar10':{
        'dataset': {
            'name': 'CIFAR10',
            'kwargs': {
                'resize':[32, 32],
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': AlexNet,
                'kwargs': {'input_dim': [3, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/CIFAR10/raw/AlexNet - 9005.pt',
            },
            'student': {
                'Class': AlexNet,
                'kwargs': {'half_size': True, 'input_dim': [3, 32, 32], 'num_classes': 10},
            },
            'distiller': {
                'Class': HintonDistiller,
                'compile_kwargs': {
                    'loss_fn_distill': nn.KLDivLoss(reduction='batchmean'),
                    'coeff_dt': 0.9,
                    'coeff_lb': 0.1,
                    'temperature':1,
                },
            }
        },
        'optim': {
            'num_epochs': 200,
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones':[100, 150], 'gamma':0.1},
            },
        }
    },
    'standardkd-cifar100':{
        'dataset': {
            'name': 'CIFAR100',
            'kwargs': {
                'resize':[32, 32],
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 100,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 5000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': ResNet_CIFAR,
                'kwargs': {'depth':32, 'skip_option':'A', 'input_dim': [3, 32, 32], 'num_classes': 100},
                'pretrained_path': './pretrained/CIFAR100/raw/ResNet32_CIFAR - 7141.pt',
            },
            'student': {
                'Class': ResNet_CIFAR,
                'kwargs': {'depth':20, 'skip_option':'A', 'input_dim': [3, 32, 32], 'num_classes': 100},
            },
            'distiller': {
                'Class': HintonDistiller,
                'compile_kwargs': {
                    'loss_fn_distill': nn.KLDivLoss(reduction='batchmean'),
                    'coeff_dt': 0.9,
                    'coeff_lb': 0.1,
                    'temperature':1,
                },
            }
        },
        'optim': {
            'num_epochs': 200,
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones':[100, 150], 'gamma':0.1},
            },
        }
    },
    'standardkd-tinyimagenet':{
        'dataset': {
            'name': 'TinyImageNet',
            'kwargs': {
                'splits': ['train', 'val'],
                'resize':[64, 64],
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 64, 64],
            'num_classes': 200,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 10000, 'balanced': True, 'shuffle':True},
            'split_remap': {'val': 'test'},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'val':False}},
            'batch_size': {'train': 128, 'val': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': ResNet_CIFAR,
                'kwargs': {'depth':32, 'skip_option':'A', 'input_dim': [3, 64, 64], 'num_classes':200},
                'pretrained_path': './pretrained/TinyImageNet/raw/ResNet32_CIFAR - 5627.pt',
            },
            'student': {
                'Class': ResNet_CIFAR,
                'kwargs': {'depth':20, 'skip_option':'A', 'input_dim': [3, 64, 64], 'num_classes':200},
            },
            'distiller': {
                'Class': HintonDistiller,
                'compile_kwargs': {
                    'loss_fn_distill': nn.KLDivLoss(reduction='batchmean'),
                    'coeff_dt': 0.9,
                    'coeff_lb': 0.1,
                    'temperature':1,
                },
            }
        },
        'optim': {
            'num_epochs': 200,
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones':[100, 150], 'gamma':0.1},
            },
        }
    },
    'standardkd-imagenette': {
        'dataset': {
            'name': 'Imagenette',
            'kwargs': {
                'init_augmentations': {
                    'train': transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                    ]),
                    'val': transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                    ]),
                },
            },
            'image_dim': [3, 224, 224],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'split_remap': {'val': 'test'},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'val':False}},
            'batch_size': {'train': 128, 'val': 128},
            'kwargs': {'num_workers': 2, 'prefetch_factor': 2, 'pin_memory': True},
        },
        'models': {
            'teacher': {
                'Class': torchvision.models.resnet34,
                'kwargs': {'num_classes': 10},
                'name': 'ResNet34',
                'pretrained_path': './pretrained/Imagenette/raw/ResNet34 - 9047.pt',
                'pin_memory': True,
            },
            'student': {
                'Class': torchvision.models.resnet18,
                'kwargs': {'num_classes': 10},
                'name': 'ResNet18',
            },
            'distiller': {
                'Class': HintonDistiller,
                'compile_kwargs': {
                    'loss_fn_distill': nn.KLDivLoss(reduction='batchmean'),
                    'coeff_dt': 0.9,
                    'coeff_lb': 0.1,
                    'temperature':1,
                },
            }
        },
        'optim': {
            'num_epochs': 200,
            'opt_S': {
                'Class': torch.optim.SGD,
                'kwargs': {'lr':1e-1, 'momentum':0.9, 'weight_decay':5e-4},
            },
            'sch_S': {
                'Class': torch.optim.lr_scheduler.MultiStepLR,
                'kwargs': {'milestones':[100, 150], 'gamma':0.1},
            },
        },
    },

    'supp-samearch-mnist': {NotImplementedError},
    'supp-samearch-fmnist': {NotImplementedError},
    'supp-samearch-svhn': {NotImplementedError},
    'supp-samearch-cifar10': {NotImplementedError},
    'supp-samearch-cifar100': {NotImplementedError},
    'supp-samearch-tinyimagenet': {NotImplementedError},
    'supp-samearch-imagenette': {NotImplementedError},

    'abla-components-wgangp-mnist': {NotImplementedError},
    'abla-components-wgangpaf-mnist': {NotImplementedError},
    'abla-components-wgangpafcb-mnist': {NotImplementedError}, # == main-mnist
    'abla-components-wgangp-fmnist': {NotImplementedError},
    'abla-components-wgangpaf-fmnist': {NotImplementedError},
    'abla-components-wgangpafcb-fmnist': {NotImplementedError}, # == main-fmnist
    'abla-components-wgangp-svhn': {NotImplementedError},
    'abla-components-wgangpaf-svhn': {NotImplementedError},
    'abla-components-wgangpafcb-svhn': {NotImplementedError}, # == main-svhn
    'abla-components-noise-cifar10': {NotImplementedError},
    'abla-components-gan-cifar10': {NotImplementedError},
    'abla-components-wgan-cifar10': {NotImplementedError},
    'abla-components-wgangp-cifar10': {NotImplementedError},
    'abla-components-wgangpaf-cifar10': {NotImplementedError},
    'abla-components-wgangpcb-cifar10': {NotImplementedError},
    'abla-components-wgangpafcb-cifar10': {NotImplementedError}, # == main-cifar10
    'abla-components-wgangp-tinyimagenet': {NotImplementedError},
    'abla-components-wgangpaf-tinyimagenet': {NotImplementedError},
    'abla-components-wgangpcb-tinyimagenet': {NotImplementedError},
    'abla-components-wgangpafcb-tinyimagenet': {NotImplementedError}, # == main-tinyimagenet


    'abla-N-divbfkd-mnist': {'main_config': 'main-mnist'},
    'abla-N-divbfkd-fmnist': {'main_config': 'main-fmnist'},
    'abla-N-divbfkd-cifar10': {'main_config': 'main-cifar10'},
    'abla-N-standardkd-mnist': {'main_config': 'standardkd-mnist'},
    'abla-N-standardkd-fmnist': {'main_config': 'standardkd-fmnist'},
    'abla-N-standardkd-cifar10': {'main_config': 'standardkd-cifar10'},
    
    'abla-M-mnist': {'main_config': 'main-mnist'},
    'abla-M-cifar10': {'main_config': 'main-cifar10'},

    'abla-q-cifar10': {NotImplementedError},

    'supp-crossarch-divbfkd-res32-alexh-cifar10': {NotImplementedError},
    'supp-crossarch-divbfkd-res32-res20-cifar10': {NotImplementedError}, # models -> teacher and student config
    'supp-crossarch-divbfkd-res32-vgg11-cifar10': {NotImplementedError},
    'supp-crossarch-divbfkd-vgg16-alexh-cifar10': {NotImplementedError},
    'supp-crossarch-divbfkd-vgg16-res20-cifar10': {NotImplementedError},
    'supp-crossarch-divbfkd-vgg16-vgg11-cifar10': {NotImplementedError},
    'supp-crossarch-divbfkd-alex-alexh-cifar10': {NotImplementedError},
    'supp-crossarch-divbfkd-alex-res20-cifar10': {NotImplementedError},
    'supp-crossarch-divbfkd-alex-vgg11-cifar10': {NotImplementedError},
    'supp-crossarch-standardkd-res32-alexh-cifar10': {NotImplementedError},
    'supp-crossarch-standardkd-res32-res20-cifar10': {NotImplementedError}, # models -> teacher and student config
    'supp-crossarch-standardkd-res32-vgg11-cifar10': {NotImplementedError},
    'supp-crossarch-standardkd-vgg16-alexh-cifar10': {NotImplementedError},
    'supp-crossarch-standardkd-vgg16-res20-cifar10': {NotImplementedError},
    'supp-crossarch-standardkd-vgg16-vgg11-cifar10': {NotImplementedError},
    'supp-crossarch-standardkd-alex-alexh-cifar10': {NotImplementedError},
    'supp-crossarch-standardkd-alex-res20-cifar10': {NotImplementedError},
    'supp-crossarch-standardkd-alex-vgg11-cifar10': {NotImplementedError},
    'supp-crossarch-studentalone-alexh-cifar10': {NotImplementedError},
    'supp-crossarch-studentalone-res20-cifar10': {NotImplementedError},
    'supp-crossarch-studentalone-vgg11-cifar10': {NotImplementedError},
    'supp-crossarch-studentfull-alexh-cifar10': {NotImplementedError},
    'supp-crossarch-studentfull-res20-cifar10': {NotImplementedError},
    'supp-crossarch-studentfull-vgg11-cifar10': {NotImplementedError},

    'abla-filter-block_all-cifar10': {NotImplementedError}, # == abla-components-wgan-cifar10
    'abla-filter-pass_all-cifar10': {NotImplementedError}, # q = 0
    'abla-filter-mean_all-cifar10': {NotImplementedError}, 
    'abla-filter-quantile_all-cifar10': {NotImplementedError},
    'abla-filter-quantile_class-cifar10': {NotImplementedError}, # == main-cifar10

    'wgan-mnist': {
        'dataset': {
            'name': 'MNIST',
            'kwargs': {'resize':[32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'train': 250, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 100, 'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
            },
            'discriminator': {
                'Class': Discriminator,
                'kwargs': {'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'return_logits': True},
            },
            'GAN': {
                'Class': WGAN_GP,
                'compile_kwargs': {
                    'loss_fn': WassersteinLoss(reduction='mean'),
                    'n_critic': 5,
                    'lambda_gp': 10,
                },
            }
        },
        'optim': {
            'num_epochs': 50,
            'opt_G': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr':5e-5},
            },
            'opt_C': {
                'Class': torch.optim.RMSprop,
                'kwargs': {'lr':5e-5},
            },
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': False,
        },
    },

    'viz-embed1d-umap-mnist': {
        'dataset': {
            'name': 'MNIST',
            'kwargs': {'resize': [32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 24000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 24000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 24000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 24000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/MNIST/raw/LeNet5 - 9928.pt',
                'feature_layer': 'flatten',
                'feature_dim': [120],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 100, 'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/MNIST/WGAN.pt',
                'pretrained_path_divbfkd': './logs/MNIST/WGAN-DivBFKD.pt',
            },
        },
        'embed': {
            'Class': UMAP,
            'kwargs': {'n_components': 1},
            'toggle_pool': {
                'trainfull': True,
                'trainfew': False,
                'mixupfew': False,
                'cvae': False,
                'wgan': False,
                'divbfkd-unfiltered': False,
                'divbfkd': True,
                'fullkd': False,
            },
        },
        'viz': {
            'save_path': './logs/MNIST/viz-embed1d-umap-mnist.pdf',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
            'kwargs': {
                'linewidth': 2,
                'alpha': 0.3,
            },
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': False,
        },
    },
    'viz-embed1d-umap-fmnist': {
        'dataset': {
            'name': 'FashionMNIST',
            'kwargs': {'resize': [32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 48000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 48000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 48000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 48000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/FMNIST/raw/LeNet5 - 9090.pt',
                'feature_layer': 'flatten',
                'feature_dim': [120],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 200, 'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/FashionMNIST/WGAN.pt',
                'pretrained_path_divbfkd': './logs/FashionMNIST/WGAN-DivBFKD.pt',
            },
        },
        'embed': {
            'Class': UMAP,
            'kwargs': {'n_components': 1},
            'toggle_pool': {
                'trainfull': True,
                'trainfew': False,
                'mixupfew': False,
                'cvae': False,
                'wgan': False,
                'divbfkd-unfiltered': False,
                'divbfkd': True,
                'fullkd': False,
            },
        },
        'viz': {    
            'save_path': './logs/FashionMNIST/viz-embed1d-umap-fmnist.pdf',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
            'kwargs': {
                'linewidth': 2,
                'alpha': 0.3,
            },
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': False,
        },
    },
    'viz-embed1d-umap-cifar10': {
        'dataset': {
            'name': 'CIFAR10',
            'kwargs': {
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 40000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': AlexNet,
                'kwargs': {'input_dim': [3, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/CIFAR10/raw/AlexNet - 9005.pt',
                'feature_layer': 'flatten',
                'feature_dim': [1152],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 256, 'image_dim': [3, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/CIFAR10/WGAN.pt',
                'pretrained_path_divbfkd': './logs/CIFAR10/WGAN-DivBFKD.pt',
            },
            'cvae': {
                'Class': load_model,
                'kwargs': {'filepath': './tests/fsbbt2_synthetic/cvae_decoder_cifar10_subset_method_input_cond_la2_bs64_ep601_teacher_original.h5'},
                'latent_dim': 2,
            },
        },
        'embed': {
            'Class': UMAP,
            'kwargs': {'n_components': 1},
            'toggle_pool': {
                'trainfull': True,
                'trainfew': False,
                'mixupfew': False,
                'cvae': False,
                'wgan': False,
                'divbfkd-unfiltered': False,
                'divbfkd': True,
                'fullkd': False,
            },
        },
        'viz': {
            'save_path': './logs/CIFAR10/viz-embed1d-umap-cifar10.pdf',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
            'kwargs': {
                'linewidth': 2,
                'alpha': 0.3,
            },
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': False,
        },
    },
    'viz-embed1d-tsne-mnist': {
        'dataset': {
            'name': 'MNIST',
            'kwargs': {'resize': [32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 24000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 24000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 24000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 24000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/MNIST/raw/LeNet5 - 9928.pt',
                'feature_layer': 'flatten',
                'feature_dim': [120],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 100, 'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/MNIST/WGAN.pt',
                'pretrained_path_divbfkd': './logs/MNIST/WGAN-DivBFKD.pt',
            },
        },
        'embed': {
            'Class': MulticoreTSNE,
            'kwargs': {
                'n_components':1,
                'perplexity':50,
                'n_iter':5000,
                'n_jobs':8,
            },
            'toggle_pool': {
                'trainfull': True,
                'trainfew': False,
                'mixupfew': False,
                'cvae': False,
                'wgan': False,
                'divbfkd-unfiltered': False,
                'divbfkd': True,
                'fullkd': False,
            },
        },
        'viz': {
            'save_path': './logs/MNIST/viz-embed1d-tsne-mnist.pdf',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
            'kwargs': {
                'linewidth': 2,
                'alpha': 0.3,
            },
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': False,
        },
    },
    'viz-embed1d-tsne-fmnist': {
        'dataset': {
            'name': 'FashionMNIST',
            'kwargs': {'resize': [32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 48000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 48000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 48000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 48000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/FMNIST/raw/LeNet5 - 9090.pt',
                'feature_layer': 'flatten',
                'feature_dim': [120],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 200, 'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/FashionMNIST/WGAN.pt',
                'pretrained_path_divbfkd': './logs/FashionMNIST/WGAN-DivBFKD.pt',
            },
        },
        'embed': {
            'Class': MulticoreTSNE,
            'kwargs': {
                'n_components':1,
                'perplexity':50,
                'n_iter':5000,
                'n_jobs':8,
            },
            'toggle_pool': {
                'trainfull': True,
                'trainfew': False,
                'mixupfew': False,
                'cvae': False,
                'wgan': False,
                'divbfkd-unfiltered': False,
                'divbfkd': True,
                'fullkd': False,
            },
        },
        'viz': {
            'save_path': './logs/FashionMNIST/viz-embed1d-tsne-fmnist.pdf',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
            'kwargs': {
                'linewidth': 2,
                'alpha': 0.3,
            },
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': False,
        },
    },
    'viz-embed1d-tsne-cifar10': {
        'dataset': {
            'name': 'CIFAR10',
            'kwargs': {
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 40000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': AlexNet,
                'kwargs': {'input_dim': [3, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/CIFAR10/raw/AlexNet - 9005.pt',
                'feature_layer': 'flatten',
                'feature_dim': [1152],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 256, 'image_dim': [3, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/CIFAR10/WGAN.pt',
                'pretrained_path_divbfkd': './logs/CIFAR10/WGAN-DivBFKD.pt',
            },
            'cvae': {
                'Class': load_model,
                'kwargs': {'filepath': './tests/fsbbt2_synthetic/cvae_decoder_cifar10_subset_method_input_cond_la2_bs64_ep601_teacher_original.h5'},
                'latent_dim': 2,
            },
        },
        'embed': {
            'Class': MulticoreTSNE,
            'kwargs': {
                'n_components':1,
                'perplexity':50,
                'n_iter':5000,
                'n_jobs':8,
            },
            'toggle_pool': {
                'trainfull': True,
                'trainfew': False,
                'mixupfew': False,
                'cvae': False,
                'wgan': False,
                'divbfkd-unfiltered': False,
                'divbfkd': True,
                'fullkd': False,
            },
        },
        'viz': {
            'save_path': './logs/CIFAR10/viz-embed1d-tsne-cifar10.pdf',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
            'kwargs': {
                'linewidth': 2,
                'alpha': 0.3,
            },
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': False,
        },
    },

    'viz-embed2d-umap-mnist': {
        'dataset': {
            'name': 'MNIST',
            'kwargs': {'resize': [32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 24000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 24000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 24000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 24000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/MNIST/raw/LeNet5 - 9928.pt',
                'feature_layer': 'flatten',
                'feature_dim': [120],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 100, 'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/MNIST/WGAN.pt',
                'pretrained_path_divbfkd': './logs/MNIST/WGAN-DivBFKD.pt',
            },
        },
        'embed': {
            'Class': UMAP,
            'kwargs': {'n_components': 2},
            'toggle_pool': {
                'trainfull': True,
                'trainfew': True,
                'mixupfew': False,
                'cvae': False,
                'wgan': True,
                'divbfkd-unfiltered': True,
                'divbfkd': True,
                'fullkd': True,
            },
        },
        'viz': {
            'save_path': './logs/MNIST/viz-embed2d-umap-mnist.png',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
            'kwargs': {'s': 3},
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': True,
        },
    },
    'viz-embed2d-umap-fmnist': {
        'dataset': {
            'name': 'FashionMNIST',
            'kwargs': {'resize': [32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 48000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 48000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 48000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 48000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/FMNIST/raw/LeNet5 - 9090.pt',
                'feature_layer': 'flatten',
                'feature_dim': [120],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 200, 'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/FashionMNIST/WGAN.pt',
                'pretrained_path_divbfkd': './logs/FashionMNIST/WGAN-DivBFKD.pt',
            },
        },
        'embed': {
            'Class': UMAP,
            'kwargs': {'n_components': 2},
            'toggle_pool': {
                'trainfull': True,
                'trainfew': True,
                'mixupfew': False,
                'cvae': False,
                'wgan': True,
                'divbfkd-unfiltered': True,
                'divbfkd': True,
                'fullkd': True,
            },
        },
        'viz': {
            'save_path': './logs/FashionMNIST/viz-embed2d-umap-fmnist.png',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
            'kwargs': {'s': 3},
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': True,
        },
    },
    'viz-embed2d-umap-cifar10': {
        'dataset': {
            'name': 'CIFAR10',
            'kwargs': {
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 2000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 40000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': AlexNet,
                'kwargs': {'input_dim': [3, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/CIFAR10/raw/AlexNet - 9005.pt',
                'feature_layer': 'flatten',
                'feature_dim': [1152],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 256, 'image_dim': [3, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/CIFAR10/WGAN.pt',
                'pretrained_path_divbfkd': './logs/CIFAR10/WGAN-DivBFKD.pt',
            },
            'cvae': {
                'Class': load_model,
                'kwargs': {'filepath': './tests/fsbbt2_synthetic/cvae_decoder_cifar10_subset_method_input_cond_la2_bs64_ep601_teacher_original.h5'},
                'latent_dim': 2,
            },
        },
        'embed': {
            'Class': UMAP,
            'kwargs': {'n_components': 2},
            'toggle_pool': {
                'trainfull': True,
                'trainfew': True,
                'mixupfew': False,
                'cvae': False,
                'wgan': True,
                'divbfkd-unfiltered': True,
                'divbfkd': True,
                'fullkd': True,
            },
        },
        'viz': {
            'save_path': './logs/CIFAR10/viz-embed2d-umap-cifar10.png',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
            'kwargs': {'s': 3},
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': True,
        },
    },
    'viz-embed2d-umap-cifar10-replotcvpr': {
        'dataset': {
            'name': 'CIFAR10',
            'kwargs': {
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 2000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 40000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': AlexNet,
                'kwargs': {'input_dim': [3, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/CIFAR10/raw/AlexNet - 9005.pt',
                'feature_layer': 'flatten',
                'feature_dim': [1152],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 256, 'image_dim': [3, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/CIFAR10/WGAN.pt',
                'pretrained_path_divbfkd': './logs/CIFAR10/WGAN-DivBFKD.pt',
            },
            'cvae': {
                'Class': load_model,
                'kwargs': {'filepath': './tests/fsbbt2_synthetic/cvae_decoder_cifar10_subset_method_input_cond_la2_bs64_ep601_teacher_original.h5'},
                'latent_dim': 2,
            },
        },
        'embed': {
            'Class': UMAP,
            'kwargs': {'n_components': 2, 'min_dist': 0.3},
            'toggle_pool': {
                'trainfull': True,
                'trainfew': False,
                'mixupfew': False,
                'cvae': False,
                'wgan': False,
                'divbfkd-unfiltered': False,
                'divbfkd': True,
                'fullkd': False,
            },
        },
        'viz': {
            'save_path': './logs/CIFAR10/viz-embed2d-umap-cifar10.png',
            'colors': ['tab:red', 'tab:blue'],
            'kwargs': {'s': 5},
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': False,
        },
    },
    'viz-embed2d-umap-cifar10-replotecml': {
        'dataset': {
            'name': 'CIFAR10',
            'kwargs': {
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 2000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 40000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': AlexNet,
                'kwargs': {'input_dim': [3, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/CIFAR10/raw/AlexNet - 9005.pt',
                'feature_layer': 'flatten',
                'feature_dim': [1152],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 256, 'image_dim': [3, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/CIFAR10/WGAN.pt',
                'pretrained_path_divbfkd': './logs/CIFAR10/WGAN-DivBFKD.pt',
            },
            'cvae': {
                'Class': load_model,
                'kwargs': {'filepath': './tests/fsbbt2_synthetic/cvae_decoder_cifar10_subset_method_input_cond_la2_bs64_ep601_teacher_original.h5'},
                'latent_dim': 2,
            },
        },
        'embed': {
            'Class': UMAP,
            'kwargs': {'n_components': 2, 'min_dist': 0.3},
            'toggle_pool': {
                'trainfull': True,
                'trainfew': True,
                'mixupfew': False,
                'cvae': False,
                'wgan': False,
                'divbfkd-unfiltered': False,
                'divbfkd': True,
                'fullkd': False,
            },
        },
        'viz': {
            'save_path': './logs/CIFAR10/viz-embed2d-umap-cifar10.png',
            'colors': ['tab:red', 'black', 'tab:blue'],
            'kwargs': {'s': 3.5},
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': False,
        },
    },
    'viz-embed2d-tsne-mnist': {
        'dataset': {
            'name': 'MNIST',
            'kwargs': {'resize': [32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 24000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 24000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 24000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 24000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/MNIST/raw/LeNet5 - 9928.pt',
                'feature_layer': 'flatten',
                'feature_dim': [120],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 100, 'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/MNIST/WGAN.pt',
                'pretrained_path_divbfkd': './logs/MNIST/WGAN-DivBFKD.pt',
            },
        },
        'embed': {
            'Class': MulticoreTSNE,
            'kwargs': {
                'n_components':2,
                'perplexity':50,
                'n_iter':5000,
                'n_jobs':8,
            },
            'toggle_pool': {
                'trainfull': True,
                'trainfew': False,
                'mixupfew': False,
                'cvae': False,
                'wgan': False,
                'divbfkd-unfiltered': False,
                'divbfkd': True,
                'fullkd': False,
            },
        },
        'viz': {
            'save_path': './logs/MNIST/viz-embed2d-tsne-mnist.png',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
            'kwargs': {'s': 3},
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': False,
        },
    },
    'viz-embed2d-tsne-fmnist': {
        'dataset': {
            'name': 'FashionMNIST',
            'kwargs': {'resize': [32, 32]},
            'image_dim': [1, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 48000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 48000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 48000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 48000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': LeNet5,
                'kwargs': {'input_dim': [1, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/FMNIST/raw/LeNet5 - 9090.pt',
                'feature_layer': 'flatten',
                'feature_dim': [120],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 200, 'image_dim': [1, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/FashionMNIST/WGAN.pt',
                'pretrained_path_divbfkd': './logs/FashionMNIST/WGAN-DivBFKD.pt',
            },
        },
        'embed': {
            'Class': MulticoreTSNE,
            'kwargs': {
                'n_components':2,
                'perplexity':50,
                'n_iter':5000,
                'n_jobs':8,
            },
            'toggle_pool': {
                'trainfull': True,
                'trainfew': False,
                'mixupfew': False,
                'cvae': False,
                'wgan': False,
                'divbfkd-unfiltered': False,
                'divbfkd': True,
                'fullkd': False,
            },
        },
        'viz': {
            'save_path': './logs/FashionMNIST/viz-embed2d-tsne-fmnist.png',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
            'kwargs': {'s': 3},
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': False,
        },
    },
    'viz-embed2d-tsne-cifar10': {
        'dataset': {
            'name': 'CIFAR10',
            'kwargs': {
                'init_augmentations': {'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])},
            },
            'image_dim': [3, 32, 32],
            'num_classes': 10,
            'onehot_label': False,
            'fewshot_kwargs': {'num_samples': 2000, 'balanced': True, 'shuffle':True},
            'mixup_kwargs': {'num_samples': 2000, 'batch_size': 1000},
            'cvae_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'wgan_kwargs': {'num_samples': 40000, 'batch_size': 1000},
            'divbfkd_kwargs': {'num_samples': 40000, 'batch_size': 1000},
        },
        'dataloader': {
            'ddp': True,
            'ddp_kwargs': {'shuffle': {'train':True, 'test':False}},
            'batch_size': {'fewshot':128, 'train': 128, 'test': 1024},
            'kwargs': {},
        },
        'models': {
            'teacher': {
                'Class': AlexNet,
                'kwargs': {'input_dim': [3, 32, 32], 'num_classes': 10},
                'pretrained_path': './pretrained/CIFAR10/raw/AlexNet - 9005.pt',
                'feature_layer': 'flatten',
                'feature_dim': [1152],
            },
            'generator': {
                'Class': Generator,
                'kwargs': {'latent_dim': 256, 'image_dim': [3, 32, 32], 'base_dim': [256, 8, 8], 'NormLayer': nn.BatchNorm2d},
                'pretrained_path_wgan': './logs/CIFAR10/WGAN.pt',
                'pretrained_path_divbfkd': './logs/CIFAR10/WGAN-DivBFKD.pt',
            },
            'cvae': {
                'Class': load_model,
                'kwargs': {'filepath': './tests/fsbbt2_synthetic/cvae_decoder_cifar10_subset_method_input_cond_la2_bs64_ep601_teacher_original.h5'},
                'latent_dim': 2,
            },
        },
        'embed': {
            'Class': MulticoreTSNE,
            'kwargs': {
                'n_components':2,
                'perplexity':50,
                'n_iter':5000,
                'n_jobs':8,
            },
            'toggle_pool': {
                'trainfull': True,
                'trainfew': False,
                'mixupfew': False,
                'cvae': False,
                'wgan': False,
                'divbfkd-unfiltered': False,
                'divbfkd': True,
                'fullkd': False,
            },
        },
        'viz': {
            'save_path': './logs/CIFAR10/viz-embed2d-tsne-cifar10.png',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
            'kwargs': {'s': 3},
        },
        'metrics': {
            'inception_score': False,
            'alpha_precision': False,
        },
    },
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddp', type=bool, default=False)
    parser.add_argument('--config', type=str, default='main-mnist')
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--var', type=int, default=0)
    parser = parser.parse_args()

    print(f'ddp: {parser.ddp}')
    print(f'config: {parser.config}')

    for run in range(parser.run, parser.run+parser.n_runs):
        if not parser.ddp:
            parse_expt(parser.config)(None, 1, CONFIG[parser.config], run, parser.var)
        else:
            import torch.multiprocessing as mp

            world_size = torch.cuda.device_count()
            print(f'world_size: {world_size}')

            setup_master()
            mp.spawn(
                fn=parse_expt(parser.config),
                args=(world_size, CONFIG[parser.config], run, parser.var),
                nprocs=world_size,
            )