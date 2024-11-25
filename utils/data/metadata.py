from typing import Any

from torchvision import datasets

from .imagenet import ImageNet
from .mnistm import MNISTM
from .synthetic_digits import SyntheticDigits
from .tiny_imagenet import TinyImageNet


METADATA:dict[str, dict[str, Any]] = {
    'MNIST':{
        'DatasetClass': datasets.MNIST,
        'num_classes':  10,
        'splits':       ['train', 'test'],
        'mean_std':     ([0.1307], [0.3081]),
        'attr_inputs':  'data',
        'attr_targets': 'targets',
        'class_names':  [str(i) for i in range(10)],
    },
    'FashionMNIST':{
        'DatasetClass': datasets.FashionMNIST,
        'num_classes':  10,
        'splits':       ['train', 'test'],
        'mean_std':     ([0.2860], [0.3530]),
        'attr_inputs':  'data',
        'attr_targets': 'targets',
        'class_names':  ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    },
    'SVHN':{
        'DatasetClass': datasets.SVHN,
        'num_classes':  10,
        'splits':       ['train', 'test', 'extra'],
        'mean_std':     ([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]),
        'attr_inputs':  'data',
        'attr_targets': 'labels',
        'class_names':  [str(i) for i in range(10)],
    },
    'MNIST-M':{
        'DatasetClass': MNISTM,
        'num_classes':  10,
        'splits':       ['train', 'test'],
        'mean_std':     ([0.4630, 0.4668, 0.4194], [0.2544, 0.2385, 0.2625]),
        'attr_inputs':  'data',
        'attr_targets': 'targets',
        'class_names':  [str(i) for i in range(10)],
    },
    'SyntheticDigits':{
        'DatasetClass': SyntheticDigits,
        'num_classes':  10,
        'splits':       ['train', 'test'],
        'mean_std':     ([0.4639, 0.4634, 0.4643], [0.3002, 0.3001, 0.3004]),
        'attr_inputs':  'data',
        'attr_targets': 'targets',
        'class_names':  [str(i) for i in range(10)],
    },
    'USPS':{
        'DatasetClass': datasets.USPS,
        'num_classes':  10,
        'splits':       ['train', 'test'],
        'mean_std':     ([0.2469], [0.2989]),
        'attr_inputs':  'data',
        'attr_targets': 'targets',
        'class_names':  [str(i) for i in range(10)],
    },
    'CIFAR10':{
        'DatasetClass': datasets.CIFAR10,
        'num_classes':  10,
        'splits':       ['train', 'test'],
        'mean_std':     ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        'attr_inputs':  'data',
        'attr_targets': 'targets',
        'class_names':  ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    },
    'CIFAR100':{
        'DatasetClass': datasets.CIFAR100,
        'num_classes':  100,
        'splits':       ['train', 'test'],
        'mean_std':     ([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762]),
        'attr_inputs':  'data',
        'attr_targets': 'targets',
    },
    'TinyImageNet':{
        'DatasetClass': TinyImageNet,
        'num_classes':  200,
        'splits':       ['train', 'val'], # 'test': not implemented
        'mean_std':     ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),         # ImageNet
        'attr_inputs':  'data',
        'attr_targets': 'targets',
    },
    'Imagenette':{
        'DatasetClass': datasets.Imagenette,
        'num_classes':  10,
        'splits':       ['train', 'val'],
        'mean_std':     ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),         # ImageNet
        'attr_inputs':  '_samples',
        'attr_targets': 'targets',
        'class_names': ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
    },
    'ImageNet':{
        'DatasetClass': ImageNet,
        'num_classes':  1000,
        'splits':       ['train', 'val'], # 'test': not implemented
        'mean_std':     ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'attr_inputs':  'samples',
        'attr_targets': 'targets',
    },
    'Flowers102':{
        'DatasetClass': datasets.Flowers102,
        'num_classes':  102,
        'splits':       ['train', 'val', 'test'],
        'mean_std':     ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),         # ImageNet
        'attr_inputs':  '_image_files',
        'attr_targets': '_labels',
    },
    'Places365':{
        'DatasetClass': datasets.Places365,
        'num_classes':  365,
        'splits':       ['train-standard', 'train-challenge', 'val'],
        'mean_std':     ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),         # ImageNet
        'attr_inputs':  'imgs',
        'attr_targets': 'targets',
    }
}