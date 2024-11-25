from typing import Sequence

from torch import nn, Tensor


class IdentityPool2d(nn.Module):
    """Apply pooling without taking the max or average values. Used for ResNet
    with `skip_option` A (e.g., ResNetCIFAR). Setting `stride` to 1 is
    equivalent to an Identity layer.

    Args:
      `stride`: Stride of the pooling. Defaults to `2`.
    """
    def __init__(self, stride:int=2):
        super().__init__()
        self.stride = stride
    
    def forward(self, input:Tensor) -> Tensor:
        return input[:, :, ::self.stride, ::self.stride]
    
    def extra_repr(self) -> str:
        params = {
            'stride': self.stride,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class ChannelPad2d(nn.Module):
    """Pad tensors on the channel dimension (index -3). Only works for NCHW or
    CHW tensors. Used for ResNet with `skip_option` A (e.g., ResNetCIFAR).
    
    Args:
      `pad`: Number of channels to pad. Can be an `int`: pad the same number of \
        channels, or a tuple of `(int, int)` to pad different numbers of        \
        channels on each side.
      `mode`: Padding mode: `'constant'` | `'reflect'`| `'replicate'` |         \
        `'circular'`. See `torch.nn.functional.pad` for more details. Defaults  \
        to `'constant'`. 
      `value`: fill value for `'constant'` padding. Defaults to `0`.
    """
    def __init__(self,
        pad:int|Sequence[int],
        mode:str='constant',
        value:float=0
    ):
        super().__init__()
        self.pad   = pad
        self.mode  = mode
        self.value = value

        if isinstance(self.pad, int):
            self.pad = [self.pad]*2
        else:
            assert len(self.pad) == 2

    def forward(self, input:Tensor) -> Tensor:
        return nn.functional.pad(
            input=input,
            pad=[0, 0, 0, 0, *self.pad],
            mode=self.mode,
            value=self.value,
        )

    def extra_repr(self) -> str:
        params = {
            'pad':   self.pad,
            'mode':  self.mode,
            'value': self.value,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class GlobalAvgPool2d(nn.Module):
    """Apply global average pooling that take the average of each feature map.
    Only works for NCHW or CHW tensors.

    Args:
    + `keepdim`: Flag to retain the H and W dimensions in the output tensor.    \
        Defaults to `False`. 
    """
    def __init__(self, keepdim:bool=False):
        super().__init__()
        self.keepdim = keepdim

    def forward(self, input:Tensor) -> Tensor:
        return input.mean(dim=[-1, -2], keepdim=self.keepdim)
    
    def extra_repr(self) -> str:
        params = {
            'keepdim': self.keepdim,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class BasicBlock(nn.Module):
    """Basic block for building ResNet.

    Args:
      `in_channels`: Number of input channels.
      `out_channels`: Number of output channels produced by the block.
      `stride`: Stride for preset convolutional layers in the block. Should be  \
        defined in the ResNet backbone. See below for details. Defaults to `1`.
      `skip_option`: Type of skip connection: `'A'`|`'B'`. Should be defined in \
        the ResNet backbone. Defaults to `'B'`.

    Notes on `skip_option`:
    + `'A'`: The shortcut still performs identity mapping, with extra zero      \
        entries padded for increasing dimensions. This skip_option introduces no\
        extra parameter; For CIFAR10 ResNet paper uses skip_option A.
    + `'B'`: The projection shortcut in Eqn.(2) is used to match dimensions     \
        (done by 1x1 convolutions). For both skip_options, when the shortcuts go\
        across feature maps of two sizes, they are performed with a stride of 2.
    """
    expansion = 1

    def __init__(self,
        in_channels:int,
        out_channels:int,
        stride:int=1,
        skip_option='B',
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.skip_option  = skip_option

        self.main_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=self.out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if skip_option == 'A':
                self.shortcut = nn.Sequential(
                    IdentityPool2d(stride=2),
                    ChannelPad2d(pad=out_channels//4),
                )
            elif skip_option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, self.expansion*self.out_channels, kernel_size=1, stride=self.stride, bias=False),
                    nn.BatchNorm2d(self.expansion*self.out_channels),
                )
        else:
            self.shortcut = nn.Identity()

        self.connect = nn.ReLU()

    def forward(self, input:Tensor) -> Tensor:
        main_x = self.main_layers(input)
        skip_x = self.shortcut(input)
        x = self.connect(main_x + skip_x)
        return x

    def extra_repr(self) -> str:
        params = {
            'in_channels':  self.in_channels,
            'out_channels': self.out_channels,
            'stride':       self.stride,
            'skip_option':  self.skip_option,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])


class ResNet_CIFAR(nn.Module):
    '''Properly implemented ResNet for CIFAR10 as described in paper [1].

    Args:
    + `depth`: Number of layers, should be a `6n+2` integer.
    + `num_classes`: Number of output nodes. Defaults to `10`.
    + `skip_option`: Type of skip connection. Defaults to `'A'`.
    + `return_logits`: Flag to choose between return logits or probability.    \
        Defaults to `True`.

    The implementation and structure of this file is hugely influenced by [2]   \
    which is implemented for ImageNet and doesn't have `skip_option` `'A'` for  \
    identity. Moreover, most of the implementations on the web is copy-paste    \
    from `torchvision`'s `resnet` and has wrong number of params.

    Notes on `skip_option`:
    + `'A'`: The shortcut still performs identity mapping, with extra zero      \
        entries padded for increasing dimensions. This skip_option introduces no\
        extra parameter; For CIFAR10 ResNet paper uses skip_option A.
    + `'B'`: The projection shortcut in Eqn.(2) is used to match dimensions     \
        (done by 1x1 convolutions). For both skip_options, when the shortcuts go\
        across feature maps of two sizes, they are performed with a stride of 2.

    Proper ResNet for CIFAR10 (for fair comparision and etc.) has following
    number of layers and parameters:
    |name      | layers | params|
    |----------|--------|-------|
    |ResNet20  |    20  | 0.27M |
    |ResNet32  |    32  | 0.46M |
    |ResNet44  |    44  | 0.66M |
    |ResNet56  |    56  | 0.85M |
    |ResNet110 |   110  |  1.7M |
    |ResNet1202|  1202  | 19.4m |
    which this implementation indeed has.

    References:
    1. Deep Residual Learning for Image Recognition - He et al., CVPR 2016.
    2. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    If you use this implementation in you work, please don't forget to mention the
    author, Yerlan Idelbayev.
    '''
    BLOCK = BasicBlock
    
    def __init__(self,
        depth:int=20,
        input_dim:Sequence[int]=[3, 32, 32],
        num_classes=10,
        skip_option:str='A',
        return_logits:bool=True,
    ):
        assert (depth - 2) % 6 == 0, '`depth` must be a `6n+2` integer.'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.skip_option = skip_option
        self.return_logits = return_logits

        self.Block = self.BLOCK
        num_blocks = (self.depth - 2)//6

        self.in_channels = 16

        self.conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[0], out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )
        self.conv_1 = self.make_residual_layer(Block=self.Block, out_channels=16, num_blocks=num_blocks, stride=1, skip_option=self.skip_option)
        self.conv_2 = self.make_residual_layer(Block=self.Block, out_channels=32, num_blocks=num_blocks, stride=2, skip_option=self.skip_option)
        self.conv_3 = self.make_residual_layer(Block=self.Block, out_channels=64, num_blocks=num_blocks, stride=2, skip_option=self.skip_option)
        self.glb_pool = GlobalAvgPool2d()
        self.logits = nn.Linear(in_features=64, out_features=self.num_classes)
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = nn.Sigmoid()
            elif self.num_classes > 1:
                self.pred = nn.Softmax(dim=1)

        self.apply(self.init_weights)
    
    def make_residual_layer(self,
        Block:BasicBlock,
        out_channels:int,
        num_blocks:int,
        stride:int,
        skip_option:str,
    ) -> nn.Sequential:
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_channels, out_channels, stride, skip_option))
            self.in_channels = out_channels*Block.expansion

        return nn.Sequential(*layers)
    
    @staticmethod
    def init_weights(module:nn.Module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)

    def forward(self, input:Tensor) -> Tensor:
        x = self.conv_0(input)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.glb_pool(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x

    @property
    def name(self) -> str:
        return f'ResNetCIFAR{self.depth}'
    
    def extra_repr(self) -> str:
        params = {
            'depth':         self.depth,
            'input_dim':     self.input_dim,
            'num_classes':   self.num_classes,
            'skip_option':   self.skip_option,
            'return_logits': self.return_logits,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])