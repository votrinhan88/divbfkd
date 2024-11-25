from typing import Sequence

from torch import nn, Tensor
import numpy as np


class AlexNet(nn.Module):
    """ImageNet classification with deep convolutional neural networks -
    Krizhevsky et al. (2012). DOI: 10.1145/3065386

    Args:
      `half_size`: Flag to choose between AlexNet or AlexNet-Half. Defaults to  \
        `False`.
      `input_dim`: Dimension of input images. Defaults to `[3, 32, 32]`.
      `num_classes`: Number of output nodes. Defaults to `10`.
      `return_logits`: Flag to choose between return logits or probability.     \
        Defaults to `True`.
    """    
    def __init__(self,
        input_dim:Sequence[int]=[3, 32, 32],
        num_classes:int=10,
        half_size:bool=False,
        return_logits:bool=True,
    ):
        assert isinstance(half_size, bool), '`half` must be of type bool.'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'
        
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.half_size = half_size
        self.return_logits = return_logits

        divisor = 2 if self.half_size is True else 1
        
        # Convolutional blocks
        ongoing_shape = self.input_dim
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[0], out_channels=48//divisor, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=48//divisor)
        )
        ongoing_shape = [48//divisor, *[(dim - 1)//2 for dim in ongoing_shape[1:]]]
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=48//divisor, out_channels=128//divisor, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=128//divisor)
        )
        ongoing_shape = [128//divisor, *[(dim - 1)//2 for dim in ongoing_shape[1:]]]
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128//divisor, out_channels=192//divisor, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=192//divisor)
        )
        ongoing_shape = [192//divisor, *ongoing_shape[1:]]
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=192//divisor, out_channels=192//divisor, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=192//divisor)
        )
        ongoing_shape = [192//divisor, *ongoing_shape[1:]]
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=192//divisor, out_channels=128//divisor, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=128//divisor)
        )
        ongoing_shape = [128//divisor, *[(dim - 1)//2 for dim in ongoing_shape[1:]]]
        self.flatten = nn.Flatten()
        ongoing_shape = [np.prod(ongoing_shape)]
        # Fully-connected layers
        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=ongoing_shape[0], out_features=512//divisor),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(num_features=512//divisor)
        )
        ongoing_shape = [512//divisor]
        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=ongoing_shape[0], out_features=256//divisor),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(num_features=256//divisor)
        )
        ongoing_shape = [256//divisor]
        self.logits = nn.Linear(in_features=ongoing_shape[0], out_features=self.num_classes)
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = nn.Sigmoid()
            elif self.num_classes > 1:
                self.pred = nn.Softmax(dim=1)

    def forward(self, input:Tensor) -> Tensor:
        x = self.conv_1(input)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x

    @property
    def name(self) -> str:
        return f'{self.__class__.__name__}{"_Half" if self.half_size is True else ""}'
    
    def extra_repr(self) -> str:
        params = {
            'input_dim'     :self.input_dim,
            'num_classes'   :self.num_classes,
            'half_size'     :self.half_size,
            'return_logits' :self.return_logits,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])