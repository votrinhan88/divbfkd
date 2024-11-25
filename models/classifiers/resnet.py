# https://github.com/akamaster/pytorch_resnet_cifar10
# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Sequence

from torch import nn, Tensor


class IdentityPool2d(nn.Module):
    """Apply pooling without taking the max or average values. Used for ResNet
    with `skip_option` A (e.g., ResNet_CIFAR).

    Setting `stride` to 1 is equivalent to an Identity layer.

    Args:
    + `stride`: Stride of the pooling. Defaults to `2`.
    """
    def __init__(self, stride:int=2):
        super().__init__()
        self.stride = stride
    
    def forward(self, x:Tensor) -> Tensor:
        return x[:, :, ::self.stride, ::self.stride]


class ChannelPad2d(nn.Module):
    """Pad tensors on the channel dimension (index -3). Only works for NCHW or
    CHW tensors.

    Used for ResNet with `skip_option` A (e.g., ResNet_CIFAR).
    
    Args:
    + `pad`: Number of channels to pad. Can be an `int`: pad the same number of \
        channels, or a tuple of `(int, int)` to pad different numbers of        \
        channels on each side.
    + `mode`: Padding mode: `'constant'` | `'reflect'`| `'replicate'` |         \
        `'circular'``. See `torch.nn.functional.pad` for more details. Defaults \
        to `'constant'`. 
    + `value`: fill value for `'constant'` padding. Defaults to `0`.
    """
    def __init__(
        self,
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

    def forward(self, x:Tensor) -> Tensor:
        return nn.functional.pad(
            input=x,
            pad=[0, 0, 0, 0, *self.pad],
            mode=self.mode,
            value=self.value,
        )


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

    def forward(self, x:Tensor) -> Tensor:
        return x.mean(dim=[-1, -2], keepdim=self.keepdim)


class BasicBlock(nn.Module):
    """Basic block for building ResNet.  

    Args:
    + `in_channels`: Number of input channels.
    + `out_channels`: Number of output channels produced by the block.
    + `stride`: Stride for preset convolutional layers in the block. Should be  \
        defined in the ResNet backbone. See below for details. Defaults to `1`.
    + `skip_option`: Type of skip connection: `'A'`|`'B'`. Should be defined in \
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

    def __init__(self, in_channels:int, out_channels:int, stride:int=1, skip_option='B'):
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

    def forward(self, x:Tensor) -> Tensor:
        main_x = self.main_layers(x)
        skip_x = self.shortcut(x)
        x = self.connect(main_x + skip_x)
        return x


class ResNet_CIFAR(nn.Module):
    '''Properly implemented ResNet-s for CIFAR10 as described in paper [1].

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

    Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
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
    1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Deep Residual Learning\
        for Image Recognition. arXiv:1512.03385
    2. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    If you use this implementation in you work, please don't forget to mention the
    author, Yerlan Idelbayev.
    '''
    BLOCK = BasicBlock
    def __init__(
        self,
        depth:int=20,
        input_dim:Sequence[int]=[3, 32, 32],
        num_classes=10,
        skip_option:str='A',
        return_logits:bool=True
    ):
        assert (depth - 2)%6 == 0, '`depth` must be a `6n+2` integer.'
        # assert input_dim[1:] == [32, 32], '`input_dim` must be of `[C, 32, 32]`.'
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
        # self.glb_pool = nn.AvgPool2d(kernel_size=8)
        # self.flatten = nn.Flatten()
        self.glb_pool = GlobalAvgPool2d()
        self.logits = nn.Linear(in_features=64, out_features=self.num_classes)
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = nn.Sigmoid()
            elif self.num_classes > 1:
                self.pred = nn.Softmax(dim=1)

        self.apply(self.init_weights)
    
    def make_residual_layer(self, Block, out_channels, num_blocks, stride, skip_option):
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

    def forward(self, x:Tensor) -> Tensor:
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.glb_pool(x)
        # x = self.flatten(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x

    @property
    def name(self) -> str:
        return f'ResNet{self.depth}_CIFAR'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--var', type=float, default=0)
    parser = parser.parse_args()

    import torch
    from torch import optim
    from torchinfo import summary
    from torchvision import transforms

    from models.classifiers import ClassifierTrainer
    from utils.callbacks import CSVLogger, ModelCheckpoint, SchedulerOnEpochTrainEnd
    from utils.data import get_dataloader
    

    def expt_summary(with_summary:bool=False):
        """Summary the ResNet_CIFAR models of depth 20 to 1202.
        
        The models shouldhave  the following number of parameters.
        | Model      | Depth | Params (M) |
        | ---------- | ----- | ---------- |
        | ResNet20   |    20 |       0.27 |
        | ResNet32   |    32 |       0.46 |
        | ResNet44   |    44 |       0.66 |
        | ResNet56   |    56 |       0.85 |
        | ResNet110  |   110 |       1.7  |
        | ResNet1202 |  1202 |      19.4  |
        """        
        def count_parameters(model:nn.Module) -> int:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        for depth in [20, 32, 44, 56, 110, 1202]:
            net = ResNet_CIFAR(depth=depth, num_classes=10)
            if depth <= 100:
                if with_summary == True:
                    summary(model=net, input_size=[100, 3, 32, 32], depth=2)
            print(f'{net.name:>16}: {count_parameters(net):>8}')
    

    def expt_cifar_svhn(run:int=0, var:float=0):
        DATASET = 'CIFAR100'        # CIFAR10, CIFAR100, SVHN
        MODEL_CLASS, MODEL_KWARGS = ResNet_CIFAR, {'depth':var, 'skip_option':'A'}

        IMAGE_DIM = [3, 32, 32]
        NUM_CLASSES = {'CIFAR10':10, 'CIFAR100':100, 'SVHN':10}[DATASET]

        AUG = {
            'CIFAR10': transforms.Compose([
                transforms.RandomCrop(size=IMAGE_DIM[1:], padding=4),
                transforms.RandomHorizontalFlip(),
            ]),
            'CIFAR100': transforms.Compose([
                transforms.RandomCrop(size=IMAGE_DIM[1:], padding=4),
                transforms.RandomHorizontalFlip(),
            ]),
            'SVHN':transforms.RandomCrop(size=IMAGE_DIM[1:], padding=4),
        }[DATASET]

        BATCH_SIZE = 128
        NUM_EPOCHS = {'CIFAR10':200, 'CIFAR100':200, 'SVHN':100}[DATASET]

        OPT, OPT_KWARGS = optim.SGD, {'lr': 0.1, 'momentum':0.9, 'weight_decay':5e-4}
        SCH, SCH_KWARGS = optim.lr_scheduler.MultiStepLR, {'milestones':[int(0.5*NUM_EPOCHS), int(0.75*NUM_EPOCHS)], 'gamma':0.1}

        print(f' {MODEL_CLASS.__name__}(**{MODEL_KWARGS}) - {DATASET} - run {run} '.center(80,'#'))

        dataloader = get_dataloader(
            dataset=DATASET,
            augmentation_train=AUG,
            resize=IMAGE_DIM[1:] if IMAGE_DIM[1:] != [32, 32] else None,
            batch_size_train=BATCH_SIZE,
        )

        net = MODEL_CLASS(**MODEL_KWARGS, num_classes=NUM_CLASSES)
        optimizer = OPT(params=net.parameters(), **OPT_KWARGS)
        scheduler = SCH(optimizer=optimizer, **SCH_KWARGS)

        trainer = ClassifierTrainer(model=net)
        trainer.compile(
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
        )

        scheduler_cb = SchedulerOnEpochTrainEnd(scheduler=scheduler)
        csv_logger = CSVLogger(
            filename=f'./logs/{DATASET}/{net.name} - run {run}.csv',
            append=True,
        )
        best_callback = ModelCheckpoint(
            target=net,
            filepath=f'./logs/{DATASET}/{net.name} - run {run}.pt',
            monitor='val_acc',
            save_best_only=True,
            save_state_dict_only=True,
        )
        trainer.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['test'],
            callbacks=[csv_logger, best_callback, scheduler_cb],
        )

        net.load_state_dict(torch.load(
            f=best_callback.filepath,
            map_location=trainer.device,
        ))
        trainer.evaluate(dataloader['test'])