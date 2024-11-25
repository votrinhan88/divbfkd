# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from copy import deepcopy
from typing import Literal, Optional, Sequence
import warnings

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import torch
from torch import nn, Tensor

from utils.callbacks import GIFMaker


def frechet_distance(
    act_1:Tensor,
    act_2:Tensor,
    as_float64:bool=True,
    recast:bool=True,
) -> Tensor:
    """Compute the Frechet distance between flattenned activations of a network
    on two different sets of inputs, as in Frechet Inception Distance (FID). Is
    symmetrical: `FD(a, b) = FD(b, a)`.
    
    $FD = \|\mu - \mu_w\|^2 + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})$

    Args:
    + `act_1`: Activations of first inputs.
    + `act_2`: Activations of second inputs.
    + `as_float64`: Flag to use dtype `torch.float64` for calculations. Defaults\
        to `True`.
    + `recast`: Flag to cast output back to `act_1`'s dtype. Defaults to `True`.

    Returns:
    + Frechet distance.
    """
    if recast == True:
        dtype = act_1.dtype
    if as_float64 == True:
        act_1 = act_1.to(dtype=torch.float64)
        act_2 = act_2.to(dtype=torch.float64)

    mu_1 = act_1.mean(dim=0)
    mu_2 = act_2.mean(dim=0)
    Sigma_1 = torch.cov(act_1.t())
    Sigma_2 = torch.cov(act_2.t())

    a = ((mu_1 - mu_2)**2).sum(dim=0)
    b = Sigma_1.trace() + Sigma_2.trace()
    c:Tensor = torch.linalg.eigvals(Sigma_1 @ Sigma_2).sqrt().real.sum(dim=-1)

    fd = a + b - 2 * c
    if recast == True:
        fd = fd.to(dtype=dtype)
    return fd


def accumulate_ema_model(ema_model:nn.Module, new_model:nn.Module, decay=0.999):
    # Create EMA-weighted model
    ema_params = {k:v for k, v in ema_model.named_parameters()}
    new_params = {k:v for k, v in new_model.named_parameters()}
    assert ema_params.keys() == new_params.keys()
    
    with torch.no_grad():
        for k in ema_params.keys():
            ema_params[k].data = decay*ema_params[k].data + (1 - decay)*new_params[k].data


class Repeat2d(nn.Module):
    """Repeats the input based on given size.

    Args:
        `repeats`: Number of times to repeat the width and height.
    
    Typically used for the discriminator in CGAN: spefically to repeat a
    multi-hot/one-hot vector to a stack of all-ones and and all-zeros image
    channels (before concatenating with real images).
    """
    NUM_REPEATS = 2
    def __init__(self, repeats:Sequence[int]):
        if any([not isinstance(item, int) for item in repeats]):
            raise TypeError(
                f"Expected a sequence of {self.NUM_REPEATS} integers, got {type(repeats)}."
            )
        if len(repeats) != self.NUM_REPEATS:
            raise ValueError(
                f"Expected a sequence of {self.NUM_REPEATS} integers, got {type(repeats)}."
            )
        super().__init__()
        self.repeats = repeats

    def forward(self, x:Tensor) -> Tensor:
        for dim in torch.arange(start=2, end=2+self.NUM_REPEATS):
            x = x.unsqueeze(dim=dim)
        return x.repeat(repeats=[1, 1, *self.repeats])
    
    def extra_repr(self) -> str:
        return f'repeats={self.repeats}'
    

class GANGIFMaker(GIFMaker):
    """Generate a GIF of GAN-synthetic images.

    Args:
    + `generator`: Generator model. Defaults to `None`, skip to parse from host.
    + `nrows`: Number of rows in subplot figure. Defaults to `5`.
    + `ncols`: Number of columns in subplot figure. Defaults to `5`.
    + `latent_dim`: Dimension of latent space. Defaults to `None`, skip to parse\
        from generator.
    + `image_dim`: Dimension of synthetic image. Defaults to `None`, skip to    \
        parse from generator.
    + `persistent_noise`: Flag to feed a fixed latent noise to the generator.  \
        Defaults to `True`.
    
    Kwargs:
    """
    def __init__(self,
        generator:Optional[nn.Module]=None,
        nrows:int=5,
        ncols:int=5,
        latent_dim:Optional[int]=None,
        image_dim:Optional[Sequence[int]]=None,
        persistent_noise:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator  = generator
        self.nrows      = nrows
        self.ncols      = ncols
        self.latent_dim = latent_dim
        self.image_dim  = image_dim
        self.persistent_noise = persistent_noise
    
    def on_train_begin(self, logs=None):
        super().on_train_begin()
        self.lazy_init()
        self.precompute_inputs()
    
    def on_epoch_end(self, epoch, logs=None):
        with torch.inference_mode():
            x_synth = self.synthesize_images()
        if epoch % self.save_freq == 0:
            self.make_figure(x_synth.to('cpu'), epoch)
    
    def on_train_end(self, logs=None):
        self.make_gif()

    def lazy_init(self):
        """Init previously inaccessible attributes in __init__(). Should be     \
        accessed during on_train_begin().
        """
        if self.generator is None:
            self.generator = self.host.generator

        if self.latent_dim is None:
            self.latent_dim = self.host.latent_dim

        if self.image_dim is None:
            self.image_dim = self.host.image_dim

    def precompute_inputs(self):
        """Pre-compute inputs to feed to the generator."""
        batch_size = self.nrows*self.ncols
        if self.seed is None:
            self.z_synth = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
        elif self.seed is not None:
            cur_seed = torch.seed()
            torch.manual_seed(self.seed)
            self.z_synth = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
            torch.manual_seed(cur_seed)

    def synthesize_images(self) -> Tensor:
        """Produce synthetic images with the generator.
        
        Returns:
        + A batch of synthetic images.
        """
        if self.persistent_noise is False:
            batch_size = self.nrows*self.ncols
            if self.seed is None:
                self.z_synth = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
            elif self.seed is not None:
                cur_seed = torch.seed()
                torch.manual_seed(self.seed)
                self.z_synth = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
                torch.manual_seed(cur_seed)
        
        self.generator.eval()
        x_synth = self.generator(self.z_synth)
        x_synth = self.postprocess_fn(x_synth)
        return x_synth
    
    def make_figure(self, images:Tensor, value:float|int):
        """Tile the images into a nice grid, then save as a figure.
        
        Args:
        + `images`: A batch of images.
        + `epoch`: Current epoch.
        """
        fig, ax = plt.subplots(constrained_layout=True, figsize=(self.ncols, 0.5 + self.nrows))
        self.modify_suptitle(figure=fig, value=value)

        x = images
        # Tile images into a grid
        # Pad 1 pixel on top row & left column to all images in batch
        x = nn.functional.pad(x, pad=(1, 0, 1, 0), value=1) # top, bottom, left, right
        x = torch.reshape(x, shape=[self.nrows, self.ncols, *x.shape[1:]])
        x = torch.concat(torch.unbind(x, dim=0), dim=2)
        x = torch.concat(torch.unbind(x, dim=0), dim=2)
        # Crop 1 pixel on top row & left column from the concatenated image
        x = x[:, 1:, 1:]
        x = x.permute(1, 2, 0)
        x = x.clamp(min=self.vmin, max=self.vmax)
        
        self.modify_axis(axis=ax)

        if self.image_dim[0] == 1:
            ax.imshow(x.squeeze(axis=-1), cmap='gray', vmin=self.vmin, vmax=self.vmax)
        elif self.image_dim[0] > 1:
            ax.imshow(x, vmin=self.vmin, vmax=self.vmax)

        fig.savefig(self.modify_savepath(value=value))
        plt.close(fig)


class CGANGIFMaker(GANGIFMaker):
    """Generate a GIF of CGAN-synthetic images.
    
    Args:
    + `target_classes`: The conditional target classes to make synthetic images,\
        also is the columns in the figure. Defaults to `None`, skip to include  \
        all classes.
    + `samples_per_class`: Number of samples per class, also is the number of   \
        rows in the figure. Defaults to `5`.
    + `num_classes`: Number of classes. Defaults to `None`, skip to parse from  \
        generator.
    + `class_names`: Sequence of name of labels, should have equal length to    \
        number of target classes. Defaults to `None`, skip to have generic      \
        `'class x'` names.
    + `onehot_label`: Flag to indicate whether the model receives one-hot or    \
        label encoded target classes. Defaults to `None`, skip to parse from    \
        generator.

    Kwargs:
    """
    def __init__(self,
        target_classes:Optional[Sequence[int]]=None,
        samples_per_class:int=5,
        num_classes:Optional[int]=None,
        class_names:Optional[Sequence[str]]=None,
        onehot_label:Optional[bool]=None,
        **kwargs,
    ):
        for k in ['nrows', 'ncols']:
            kwargs.pop(k, None)
        
        super().__init__(**kwargs)
        self.target_classes = target_classes
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.class_names = class_names
        self.onehot_label = onehot_label

    def lazy_init(self):
        super().lazy_init()
        if self.num_classes is None:
            self.num_classes:int = self.host.num_classes

        if self.class_names is None:
            self.class_names = [f'Class {i}' for i in range(self.num_classes)]

        if self.onehot_label is None:
            self.onehot_label:bool = self.host.onehot_label

        if self.target_classes is None:
            self.target_classes = [label for label in range(self.num_classes)]
        
        self.nrows = self.samples_per_class
        self.ncols = len(self.target_classes)

    def precompute_inputs(self):
        super().precompute_inputs()

        self.label = torch.tensor(self.target_classes, dtype=torch.long, device=self.device).repeat([self.nrows])
        if self.onehot_label is True:
            self.label = nn.functional.one_hot(input=self.label, num_classes=self.num_classes)
        self.label = self.label.to(dtype=torch.float)

    def synthesize_images(self) -> Tensor:
        if self.persistent_noise is False:
            batch_size = self.nrows*self.ncols
            if self.seed is None:
                self.z_synth = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
            elif self.seed is not None:
                cur_seed = torch.seed()
                torch.manual_seed(self.seed)
                self.z_synth = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
                torch.manual_seed(cur_seed)
        
        self.generator.eval()
        with torch.inference_mode():
            x_synth = self.generator(self.z_synth, self.label)
            x_synth = self.postprocess_fn(x_synth)
        return x_synth

    def modify_axis(self, axis:Axes):
        xticks = (self.image_dim[1] + 1)*torch.arange(len(self.target_classes)) + self.image_dim[1]/2
        xticklabels = [self.class_names[label] for label in self.target_classes]
        
        axis.set_frame_on(False)
        axis.tick_params(axis='both', length=0)
        axis.set(yticks=[], xticks=xticks, xticklabels=xticklabels)


class ProGANGIFMaker(GANGIFMaker):
    def __init__(self,
        num_steps:Optional[int]=None,
        fixed_step:Optional[int]=None,
        fixed_alpha:Optional[float]=None,
        use_ema:bool=True,
        ema_decay:float=0.999,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_steps   = num_steps
        self.fixed_step  = fixed_step
        self.fixed_alpha = fixed_alpha
        self.use_ema     = use_ema
        self.ema_decay   = ema_decay
        
        self.progan_init = False
    
    def on_train_begin(self, logs=None):
        if not self.progan_init:
            super().on_train_begin(logs)
            self.progan_init = True
    
    def on_train_batch_end(self, batch: int, logs: dict = None):
        if self.use_ema:
            accumulate_ema_model(ema_model=self.ema_generator, new_model=self.generator, decay=self.ema_decay)
        return super().on_train_batch_end(batch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        self.step = logs['step']
        return super().on_epoch_end(epoch, logs)
        
    def on_train_end(self, logs=None):
        if (logs['step'] == (self.num_steps - 1)) & (logs['progressing'] == 0):
            super().on_train_end(logs)

    def lazy_init(self):
        super().lazy_init()
        if self.use_ema:
            self.ema_generator = deepcopy(self.generator)
        
        if self.num_steps is None:
            self.num_steps:int = self.host.num_steps

    def synthesize_images(self) -> Tensor:
        if self.persistent_noise is False:
            batch_size = self.nrows*self.ncols
            if self.seed is None:
                self.z_synth = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
            elif self.seed is not None:
                cur_seed = torch.seed()
                torch.manual_seed(self.seed)
                self.z_synth = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
                torch.manual_seed(cur_seed)
        
        step = self.step if self.fixed_step is None else self.fixed_step
        alpha = 1 if self.fixed_alpha is None else self.fixed_alpha

        self.generator.eval()
        with torch.inference_mode():
            if self.use_ema:
                x_synth = self.ema_generator(self.z_synth, step=step, alpha=alpha)
            else:
                x_synth = self.generator(self.z_synth, step=step, alpha=alpha)
            x_synth = self.postprocess_fn(x_synth)
        return x_synth
    
    def modify_suptitle(self, figure:Figure, value:int):
        figure.suptitle(f'{self.host.__class__.__name__} - Step {self.step} - Epoch {value}')
    

class CGANClassInterpolatedGIFMaker(GANGIFMaker):
    """Callback to generate synthetic images, interpolating between the given
    classes of a Conditional Generative Adversarial Network. The callback can
    only work with models receiving one-hot encoded inputs. It will begin work
    at the end of the last epoch.
        
    Args:
        `filename`: Path to save GIF to.
        `start_classes`: Classes at the start of interpolation along the rows. \
            Defaults to `None`, skip to include all classes.
        `stop_classes`: Classes at the stop of interpolation along the columns.\
            Defaults to `None`, skip to include all classes.
        `num_itpl`: Number of interpolation. Defaults to `51`.
        `itpl_method`: `'linspace'` for linear and `'slerp'` for spherical     \
            linear interpolation. Defaults to `'linspace'`.
        `postprocess_fn`: Post-processing function to map synthetic images back\
            to the plot range, ideally [0, 1]. Leave as `None` to skip         \
            post-processing. Defaults to `None`, skip to skip post-processing.
        `normalize`: Flag to enable plt.imshow() automatic normalization.      \
            Defaults to `True`.
        `latent_dim`: Dimension of latent space. Defaults to `None`, skip to   \
            parse from generator.
        `image_dim`: Dimension of synthetic image. Defaults to `None`, skip to \
            parse from generator.
        `num_classes`: Number of classes. Defaults to `None`, skip to parse    \
            from generator.
        `class_names`: Sequence of name of labels, should have equal length to \
            number of target classes. Defaults to `None`, skip to have generic \
            `'class x'` names.
        `persistent_noise`: Flag to feed the same latent noise to generator for the  \
            whole training. Defaults to `True`.
        `seed`: Seed to ensure reproducibility between different runs. Defaults\
            to `None`.
        `keep_last`: Flag to save last generated image. Defaults to `False`.
        `delete_png`: Flag to delete PNG files and folder at `filename/png`    \
            after training. Defaults to `True`.
        `duration`: Duration of the generated GIF in milliseconds. Defaults to \
            `5000`.
    """
    def __init__(self,
        start_classes:Sequence[int]=None,
        stop_classes:Sequence[int]=None,
        num_itpl:int=11,
        itpl_method:Literal['linspace', 'slerp']='linspace',
        num_classes:Optional[int]=None,
        class_names:Optional[Sequence[str]]=None,
        **kwargs
    ):
        assert num_itpl > 2, (
            '`num_interpolate` (including the left and right classes) must be' +
            ' larger than 2.'
        )
        assert itpl_method in ['linspace', 'slerp'], (
            "`itpl_method` must be 'linspace' or 'slerp'"
        )
        for k in ['nrows', 'ncols']:
            kwargs.pop(k, None)

        super().__init__(**kwargs)
        self.itpl_method = itpl_method
        self.start_classes = start_classes
        self.stop_classes = stop_classes
        self.num_itpl = num_itpl
        self.num_classes = num_classes
        self.class_names = class_names
        # Nullify unused inherited attributes
        self.save_freq = None

    def on_epoch_end(self, epoch, logs=None):
        # Nullify unused MakeSyntheticGIFCallback.on_epoch_end()
        pass
    
    def on_train_end(self, logs=None):
        # Interpolate from start- to stop-classes
        with torch.inference_mode():
            itpl_ratios = torch.linspace(start=0, end=1, steps=self.num_itpl, dtype=torch.float).numpy().tolist()
            for ratio in itpl_ratios:
                label = self._interpolate(start=self.start, stop=self.stop, ratio=ratio)
                self.label = torch.cat(torch.unbind(label, dim=0), dim=0).to(self.device)
                x_synth = self.synthesize_images()
                self.make_figure(x_synth.to('cpu'), ratio)

        # Make GIF
        self.make_gif()

    def lazy_init(self):
        super().lazy_init()

        if self.host.onehot_label is None:
            warnings.warn(
                f'Host does not have attribute `onehot_label`. ' +
                'Proceed with assumption that it receives one-hot encoded inputs.')
            self.onehot_label = True
        elif self.host.onehot_label is not None:
            assert self.host.onehot_label is True, (
                'Callback only works with models receiving one-hot encoded inputs.'
            )
            self.onehot_label = True

        if self.num_classes is None:
            self.num_classes:int = self.host.num_classes

        if self.class_names is None:
            self.class_names = [f'Class {i}' for i in range(self.num_classes)]

        # Parse interpolate method, start_classes and stop_classes
        if self.itpl_method == 'linspace':
            self._interpolate = self.linspace
        elif self.itpl_method == 'slerp':
            self._interpolate = self.slerp

        if self.start_classes is None:
            self.start_classes = [label for label in range(self.num_classes)]
        if self.stop_classes is None:
            self.stop_classes = [label for label in range(self.num_classes)]
        self.start_classes = torch.tensor(self.start_classes, dtype=torch.long)
        self.stop_classes = torch.tensor(self.stop_classes, dtype=torch.long)

        self.nrows = self.start_classes.shape[0]
        self.ncols = self.stop_classes.shape[0]

    def precompute_inputs(self):
        super().precompute_inputs()
        # Convert to one-hot labels
        start = nn.functional.one_hot(input=self.start_classes, num_classes=self.num_classes).to(dtype=torch.float, device=self.device)
        stop = nn.functional.one_hot(input=self.stop_classes, num_classes=self.num_classes).to(dtype=torch.float, device=self.device)

        # Expand dimensions to have shape [nrows, ncols, num_classes]
        start = torch.unsqueeze(input=start, dim=1)
        start = torch.repeat_interleave(input=start, repeats=self.ncols, dim=1)
        stop = torch.unsqueeze(input=stop, dim=0)
        stop = torch.repeat_interleave(input=stop, repeats=self.nrows, dim=0)

        self.start = start
        self.stop = stop

        if self.itpl_method == 'slerp':
            # Normalize (L2) to [-1, 1] for numerical stability
            norm_start = start/torch.linalg.norm(start, axis=-1)
            norm_stop = stop/torch.linalg.norm(stop, axis=-1)

            dotted = (norm_start*norm_stop).sum(axis=-1)
            # Clip to [-1, 1] for numerical stability
            clipped = torch.clamp(dotted, -1, 1)
            omegas = torch.acos(clipped)
            sinned = torch.sin(omegas)

            # Expand dimensions to have shape [nrows, ncols, num_classes]
            omegas = torch.unsqueeze(input=omegas, dim=-1)
            omegas = torch.repeat_interleave(input=omegas, repeats=self.num_classes, dim=-1)
            sinned = torch.unsqueeze(input=sinned, dim=-1)
            sinned = torch.repeat_interleave(input=sinned, repeats=self.num_classes, dim=-1)
            zeros_mask = (omegas == 0)

            self.omegas = omegas
            self.sinned = sinned
            self.zeros_mask = zeros_mask

    def synthesize_images(self) -> Tensor:
        if self.persistent_noise is False:
            batch_size = self.nrows*self.ncols
            if self.seed is None:
                self.z_synth = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
            elif self.seed is not None:
                cur_seed = torch.seed()
                torch.manual_seed(self.seed)
                self.z_synth = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
                torch.manual_seed(cur_seed)
        
        self.generator.eval()
        with torch.inference_mode():
            x_synth = self.generator(self.z_synth, self.label)
            x_synth = self.postprocess_fn(x_synth)
        return x_synth

    def modify_suptitle(self, figure:Figure, value:float) -> str:
        figure.suptitle(f'{self.host.__class__.__name__} - {self.itpl_method} interpolation: {value*100:.2f}%')

    def modify_axis(self, axis:Axes):
        xticks = (self.image_dim[2] + 1)*np.arange(len(self.stop_classes)) + self.image_dim[2]/2
        xticklabels = [self.class_names[label] for label in self.stop_classes]

        yticks = (self.image_dim[1] + 1)*np.arange(len(self.start_classes)) + self.image_dim[1]/2
        yticklabels = [self.class_names[label] for label in self.start_classes]
        
        axis.set_frame_on(False)
        axis.tick_params(axis='both', length=0)
        axis.set(
            xlabel='Stop classes', xticks=xticks, xticklabels=xticklabels,
            ylabel='Start classes', yticks=yticks, yticklabels=yticklabels
        )

    def modify_savepath(self, value:float) -> str:
        return f"{self.path_png_folder}/{self.host.__class__.__name__}_itpl_{value:.4f}.png"

    def linspace(self, start:float, stop:float, ratio:float) -> float:
        label = ((1-ratio)*start + ratio*stop)
        return label
    
    def slerp(self, start:float, stop:float, ratio:float) -> Tensor:
        label = torch.where(
            condition=self.zeros_mask,
            # Normal case: omega(s) != 0
            input=self.linspace(start=start, stop=stop, ratio=ratio),
            # Special case: omega(s) == 0 --> Use L'Hospital's rule for sin(0)/0
            other=(
                torch.sin((1-ratio)*self.omegas) / self.sinned * start
                + torch.sin(ratio  *self.omegas) / self.sinned * stop
            )
        )
        return label


class AffineFunction(nn.Module):
    def __init__(self,
        from_range:Sequence[float],
        to_range:Sequence[float]=[0, 1],
    ):
        super().__init__()
        self.from_range = from_range
        self.to_range   = to_range
        
        self.m = (to_range[1] - to_range[0])/(from_range[1] - from_range[0])
        self.c = to_range[0] - (to_range[1] - to_range[0])*from_range[0]/(from_range[1] - from_range[0])
    
    def forward(self, input:Tensor) -> Tensor:
        x = self.m*input + self.c
        return x
    
    def extra_repr(self) -> str:
        params = {
            'from_range': self.from_range,
            'to_range':   self.to_range,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class AffineFunctionAdaptive(nn.Module):
    _to_range: Tensor
    
    def __init__(self,
        affine_dim:int,
        to_range:Sequence[float]=[0, 1],
    ):
        super().__init__()
        self.affine_dim = affine_dim
        self.to_range = to_range
        self.register_buffer(
            name='_to_range', 
            tensor=torch.tensor(self.to_range), 
            persistent=False,
        )

    def forward(self, input:Tensor) -> Tensor:
        x_flat = input.flatten(start_dim=self.affine_dim, end_dim=-1)
        # Compute min, max
        x_min = x_flat.min(dim=self.affine_dim)[0]
        x_max = x_flat.max(dim=self.affine_dim)[0]
        for _ in range(len(input.shape) - self.affine_dim):
            x_min = x_min.unsqueeze(dim=-1)
            x_max = x_max.unsqueeze(dim=-1)
        from_range = torch.stack(tensors=[x_min, x_max], dim=0)
        
        # Shift from [min, max] to [0, 1]
        to_range = self._to_range.reshape(2, *[1]*len(input.shape))
        m = (to_range[1] - to_range[0])/(from_range[1] - from_range[0])
        c = to_range[0] - (to_range[1] - to_range[0])*from_range[0]/(from_range[1] - from_range[0])
        x = m*input + c
        return x
    
    def extra_repr(self) -> str:
        params = {
            'affine_dim': self.affine_dim,
            'to_range':   self.to_range,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])