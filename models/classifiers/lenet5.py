from typing import Sequence

from torch import nn, Tensor


class LeNet5(nn.Module):
    '''Gradient-based learning applied to document recognition. DOI:
    10.1109/5.726791.

    Args:
      `input_dim`: Dimension of input images. Defaults to `[1, 32, 32]`.
      `num_classes`: Number of output nodes. Defaults to `10`.
      `half`: Flag to choose between LeNet-5 or LeNet-5-Half. Defaults to       \
        `False`.
      `return_logits`: Flag to choose between return logits or probability.     \
        Defaults to `True`.
    '''
    def __init__(self,
        input_dim:Sequence[int]=[1, 32, 32],
        num_classes:int=10,
        half_size:bool=False,
        return_logits:bool=True,
    ):
        assert isinstance(half_size, bool), '`half_size` must be of type bool'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'
        
        super().__init__()
        self.half_size = half_size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.return_logits = return_logits

        divisor = 2 if self.half_size is True else 1
        
        # Layers: C: convolutional, A: activation, S: pooling
        self.C1 = nn.Conv2d(in_channels=self.input_dim[0], out_channels=6//divisor, kernel_size=5, stride=1, padding=0)
        self.A1 = nn.ReLU()
        self.S2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.C3 = nn.Conv2d(in_channels=6//divisor, out_channels=16//divisor, kernel_size=5, stride=1, padding=0)
        self.A3 = nn.ReLU()
        self.S4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.C5 = nn.Conv2d(in_channels=16//divisor, out_channels=120//divisor, kernel_size=5, stride=1, padding=0)
        self.A5 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.F6 = nn.Linear(in_features=120//divisor, out_features=84//divisor)
        self.A6 = nn.ReLU()
        self.logits = nn.Linear(in_features=84//divisor, out_features=self.num_classes)
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = nn.Sigmoid()
            elif self.num_classes > 1:
                self.pred = nn.Softmax(dim=1)

    def forward(self, input:Tensor) -> Tensor:
        x = self.C1(input)
        x = self.A1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.A3(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.A5(x)
        x = self.flatten(x)
        x = self.F6(x)
        x = self.A6(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x
    
    @property
    def name(self) -> str:
        return f'{self.__class__.__name__}{"_Half" if self.half_size is True else ""}'
    
    def extra_repr(self) -> str:
        params = {
            'input_dim'    : self.input_dim,
            'num_classes'  : self.num_classes,
            'half_size'    : self.half_size,
            'return_logits': self.return_logits,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])