from ._gan import _GAN
from .dcgan import DCGAN
from .wgan import WGAN, WeightClamper
from .wgan_gp import WGAN_GP
from .losses import (
    WassersteinLoss, SoftplusLoss,
    R1RegularizationLoss, GradientPenalty1LipschitzLoss,
)
from .utils import (
    frechet_distance, accumulate_ema_model, Repeat2d,
    GANGIFMaker, CGANGIFMaker, ProGANGIFMaker, CGANClassInterpolatedGIFMaker,
    AffineFunction, AffineFunctionAdaptive
)

del _gan
del dcgan
del wgan
del wgan_gp
del losses
del utils