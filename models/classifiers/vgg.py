from typing import Optional, Sequence

from torch import nn, Tensor
from numpy import prod


class VGG(nn.Module):
    """Args:
    + `ver`: Defaults to `11`.
    + `input_dim`: Defaults to `[3, 224, 224]`.
    + `num_classes`: Defaults to `1000`.
    + `NormLayer`: Defaults to `None`.
    + `dropout`: Defaults to `0.5`.
    + `half_size`: Defaults to `False`.
    + `init_weights`: Defaults to `True`.
    + `return_logits`: Defaults to `True`.
    https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html
    """
    ver_depth:dict[str|int] = {"A":11, "B":13, "D":16, "E":19}
    config:dict[int|str, Sequence[int|str]] = {
        11 : [[64, "M"], [128, "M"], [256, 256, "M"], [512, 512, "M"], [512, 512, "M"]],
        13 : [[64, 64, "M"], [128, 128, "M"], [256, 256, "M"], [512, 512, "M"], [512, 512, "M"]],
        16 : [[64, 64, "M"], [128, 128, "M"], [256, 256, 256, "M"], [512, 512, 512, "M"], [512, 512, 512, "M"]],
        19 : [[64, 64, "M"], [128, 128, "M"], [256, 256, 256, 256, "M"], [512, 512, 512, 512, "M"], [512, 512, 512, 512, "M"]],
    }

    def __init__(self,
        ver:int|str=11,
        input_dim:Sequence[int]=[3, 224, 224],
        num_classes:int=1000,
        NormLayer:Optional[nn.Module]=None,
        dropout:float=0.5,
        half_size:bool=False,
        init_weights:bool = True,
        return_logits:bool=True,
    ):
        super().__init__()
        self.ver           = self.ver_depth.get(ver, ver)
        self.input_dim     = input_dim
        self.num_classes   = num_classes
        self.NormLayer     = NormLayer
        self.dropout       = dropout
        self.half_size     = half_size
        self.return_logits = return_logits
        if self.half_size is False:
            divisor = 1
        elif self.half_size is True:
            divisor = 2
                    
        cur_channels = input_dim[0]
        blocks = []
        for block in self.config[self.ver]:
            layers = []
            for layer in block:
                if layer == "M":
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    layers.append(nn.Conv2d(in_channels=cur_channels, out_channels=layer//divisor, kernel_size=3, padding=1))
                    cur_channels = layer//divisor
                    if self.NormLayer != None:
                        layers.append(self.NormLayer(num_features=cur_channels))
                    layers.append(nn.ReLU())
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.Sequential(*blocks)
        cur_dim = [cur_channels, *[i//(2**len(self.blocks)) for i in self.input_dim[1:]]]
        self.avgpool = nn.AdaptiveAvgPool2d(cur_dim[1:])
        self.flatten = nn.Flatten()
        # Fully-connected layers
        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=prod(cur_dim)//divisor, out_features=4096//divisor),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=4096//divisor, out_features=4096//divisor),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        self.logits = nn.Linear(in_features=4096//divisor, out_features=num_classes)
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = nn.Sigmoid()
            elif self.num_classes > 1:
                self.pred = nn.Softmax(dim=1)

        if init_weights == True:
            self.init_weights()
            
    def forward(self, input:Tensor) -> Tensor:
        x = self.blocks(input)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    @property
    def name(self) -> str:
        return f'{self.__class__.__name__}{self.ver}{"_Half" if self.half_size is True else ""}'

    def extra_repr(self) -> str:
        params = {
            'ver'          :self.ver,
            'input_dim'    :self.input_dim,
            'num_classes'  :self.num_classes,
            'NormLayer'    :self.NormLayer,
            'dropout'      :self.dropout,
            'half_size'    :self.half_size,
            'return_logits':self.return_logits,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])