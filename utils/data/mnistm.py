import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.io import loadmat
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity


class MNISTM(VisionDataset):
    filename = 'mnistm_with_label.mat'
    md5 = 'd9e18c5a88f1fe7f7ed0ea52b782e625'
    classes = [
        '0 - zero',
        '1 - one',
        '2 - two',
        '3 - three',
        '4 - four',
        '5 - five',
        '6 - six',
        '7 - seven',
        '8 - eight',
        '9 - nine',
    ]
    
    def __init__(self,
        root:str,
        split:str='train',
        transform:Optional[Callable]=None,
        target_transform:Optional[Callable]=None,
    ):
        if split not in ['train', 'test']:
            raise NotImplementedError(f"Split '{split}' is not implemented.")
        
        super(MNISTM, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.mat_file =  os.path.join(root, self.filename)
        
        if not check_integrity(fpath=self.mat_file, md5=self.md5):
            raise RuntimeError(f'`{self.filename}` not found or corrupted. Please check docstring on how to download it.')

        data = loadmat(self.mat_file)
        self.data = data[self.split]
        self.targets = np.nonzero(data[f'label_{split}'])[1]
        
    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.data)
    
    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def extra_repr(self) -> str:
        return f'Split: {self.split.capitalize()}'