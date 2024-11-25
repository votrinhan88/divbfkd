import os
import re
from typing import Sequence

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import verify_str_arg

class ImageNet(ImageFolder):
    """ImageNet ILSVRC 2012-2017 Classification Dataset.
    Link: https://image-net.org/challenges/LSVRC/index.php

    Args:
    + `root`: Root directory of the ImageNet Dataset.
    + `split`: The dataset split:  `'train'` | `'val'`. Defaults to `'train'`.
    + `short_label`: Flag to use short labels instead of ImageNet synsets.      \
        Defaults to `True`.
        
    Kwargs:
    + `transform`: A function/transform that  takes in an PIL image and returns \
        a transformed version, e.g, `transforms.RandomCrop()`. Defaults to      \
        `None`.
    + `target_transform`: A function/transform that takes in the target and     \
        transforms it. Defaults to `None`.
    + `loader`: A function to load an image given its path. Defaults to         \
        `default_loader`.
    + `is_valid_file`: A function that takes path of an Image file and check if \
        the file is a valid file (used to check of corrupt files). Defaults to  \
        `None`.

    Attributes:
    + `classes`: List of the labels if `short_label` is `True`, else list of the\
        synset tuples.
    + `class_to_idx`: Dict with items (class_name, class_index).
    + `wnids`: List of the WordNet IDs.
    + `wnid_to_idx`: Dict with items (wordnet_id, class_index).
    + `synsets`: List of the synset tuples.
    + `samples`: List of (image path, class_index) tuples. Alias: imgs.
    + `targets`: The class_index value for each image in the dataset.
    
    .. note::
        This script has been modified from PyTorch's implementation.
        Link: https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html
        Assuming the integrity of already-extracted files (by Hung Tran),
        PyTorch's `parse_archives` method is skipped. This will skip:
        + Integrity check of archives, e.g. the devkit and the data (comparing
            MD5 hash of the original and on-disk files, including a generated
            `.bin` metadata archive)
        + Extraction and parsing of the devkit and metadata (class names, etc.)
        + Extraction and parsing of the data archives (train, val)

    .. Dataset statistics (from ILSVRC 2017 devkit):
        + Training:
            + 1281167 images, with between 732 and 1300 per synset
            + bounding box annotations for at least 100 (and often many more)
                images from each synset
        + Validation:
            + 50000 images, at 50 images per category, with bounding box
                annotations for the target category
        + Test:
            + 100000 images, at 100 images per category, with bounding box
                annotations for the target category
            + The ground truth annotations will not be released.
    
    .. Metadata:
        + wnids: n01440764, n01443537, ..., n15075141
        + synsets:
            ('tench', 'Tinca tinca'),
            ('goldfish', 'Carassius auratus'),
            ...,
            ('toilet tissue', 'toilet paper', 'bathroom tissue')
        + classes:
            + if short_label == True: use short version of synsets
                'tench', 'goldfish', ..., 'toilet_tissue'
            + elif short_label == False: use synsets
            + Due to duplicating/unclear synsets, six classes are overwritten
            with PATCH_WNID_TO_CLASS.
                + n02012849 crane
                + n03126707 crane
                + n02113186 Cardigan, Cardigan Welsh corgi
                + n02963159 cardigan
                + n03710637 maillot
                + n03710721 maillot, tank suit

    .. File structure:
        (archive) imagenet-object-localization-challenge.zip
        (Hung)    class_to_name.json
        (Kaggle)  LOC_sample_submission.csv
        (Kaggle)  LOC_synset_mapping.txt
        (Kaggle)  LOC_train_solution.csv
        (Kaggle)  LOC_val_solution.csv
        (root)    ILSVRC:
        ├── Annotations/CLS-LOC
        │   ├── train
        │   │   ├── n01440764
        │   │   │   ├── n01440764_10040.xml
        │   │   │   ├── n01440764_10048.xml
        │   │   │   ├── ...
        │   │   │   ├── n01440764_9973.xml
        │   │   ├── n01443537
        │   │   ├── n01484850
        │   │   ├── ...
        │   │   ├── n15075141
        │   ├── val
        │   │   ├── ILSVRC2012_val_00000001.xml
        │   │   ├── ...
        │   │   ├── ILSVRC2012_val_00000002.xml
        │   │   ├── ILSVRC2012_val_00050000.xml
        ├── Data/CLS-LOC
        │   ├── test
        │   │   ├── ILSVRC2012_test_00000001.JPEG
        │   │   ├── ILSVRC2012_test_00000002.JPEG
        │   │   ├── ...
        │   │   ├── ILSVRC2012_test_00100000.JPEG
        │   ├── train
        │   │   ├── n01440764
        │   │   │   ├── n01440764_10026.JPEG
        │   │   │   ├── n01440764_10027.JPEG
        │   │   │   ├── ...
        │   │   │   ├── n01440764_9981.JPEG
        │   │   ├── n01443537
        │   │   ├── ...
        │   │   ├── n15075141
        │   ├── val
        │   │   ├── n01440764
        │   │   │   ├── ILSVRC2012_val_00000293.JPEG
        │   │   │   ├── ILSVRC2012_val_00002138.JPEG
        │   │   │   ├── ...
        │   │   │   ├── ILSVRC2012_val_00048969.JPEG
        │   │   ├── n01443537
        │   │   ├── n15075141
        ├── ImageSets/CLS-LOC
        │   ├── test.txt    
        │   ├── train_cls.txt
        │   ├── train_loc.txt
        │   ├── val.txt
    
    .. Devlog:
        + ----------: Hung downloaded ImageNet from Kaggle and extracted onto
            A2I2 weka folder. Link: https://www.kaggle.com/c/imagenet-object-localization-challenge
        + 2023-11-01: Nhan requested access Dataset from Hung.
        + 2023-11-03: Nhan wrote this script.
    """

    # Patch the duplicated labels: 'crane', 'cardigan', 'maillot'
    PATCH_WNID_TO_CLASS = {
        'n02012849':'crane_bird',
        'n03126707':'crane_machine',
        'n02113186':'cardigan_dog',
        'n02963159':'cardigan_clothes',
        'n03710637':'maillot',
        'n03710721':'tank_suit',
    }

    def __init__(
        self,
        root:str,
        split:str="train",
        short_label:bool=True,
        **kwargs,
    ):
        if split in ['train', 'val']:
            split_path = f'Data/CLS-LOC/{split}/'
        elif split == 'test':
            raise NotImplementedError('Test split is not yet implementeded.')

        split = verify_str_arg(
            value=split,
            arg="split",
            valid_values=['train', 'val'],
        )

        super().__init__(root=os.path.join(root, split_path), **kwargs)
        self.root = root
        self.split = split
        self.split_path = split_path
        self.short_label = short_label
        
        self.wnids = sorted(self.classes)
        self.wnid_to_idx = {v:i for (i, v) in enumerate(self.wnids)}

        self.wnid_to_synset = {}
        with open(os.path.join(self.root, '../LOC_synset_mapping.txt')) as f:
            for l in f.readlines():
                wnid, synset = re.split(pattern=' ', string=l.strip(), maxsplit=1)
                self.wnid_to_synset.update({wnid:tuple(synset.split(','))})
        self.synsets = [self.wnid_to_synset[wnid] for wnid in self.wnids]

        if self.short_label == True:
            def shorten(classes:Sequence[str]) -> str:
                label = '_'.join(classes[0].lower().split(' '))
                label = re.sub('-', '_', label)
                label = re.sub("'", '', label)
                return label
            self.classes = [shorten(s) for s in self.synsets]
        else:
            self.classes = self.synsets.copy()
            
        for (wnid, clss) in self.PATCH_WNID_TO_CLASS.items():
            self.classes[self.wnid_to_idx[wnid]] = (clss,) if self.short_label is False else clss
        self.class_to_idx = {v:i for (i, v) in enumerate(self.classes)}
        
    def extra_repr(self) -> str:
        params = {
            'root':self.root,
            'split':self.split,
            'short_label':self.short_label,
        }
        return ', '.join([f'{k}={v}' for k, v in params.items() if v is not None])

    @property
    def num_classes(self):
        return len(self.wnids)

if __name__ == '__main__':
    # Change path
    import sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    def smoke_test_imagenet():
        ROOT = '../../../weka/DataSets/imagenet_ILSVRC2017/ILSVRC'
        imagenet = ImageNet(
            root=ROOT,
            split='train',
            short_label=True,
            transform=None,
            target_transform=None,
        )

        print('SMOKE TEST IMAGENET:')
        # 1000 distinctive wnids/labels: check for short_label in [True, False]
        print(f'1000 WordNet IDs?'.ljust(40), len(set(imagenet.wnids)) == 1000)
        print(f'1000 classes?'.ljust(40), len(set(imagenet.classes)) == 1000)
        
        # Mapping from wnids/labels to digits 0-999:
        print(f'WordNet IDs map to digits 0-999?'.ljust(40), set(imagenet.wnid_to_idx.values()) == set([i for i in range(1000)]))
        print(f'Classes map to digits 0-999?'.ljust(40), set(imagenet.class_to_idx.values()) == set([i for i in range(1000)]))

        # 999 unique synsets (duplicating 'crane'): check for short_label in [True, False]
        print(f'999 unique synsets?'.ljust(40), len(set(imagenet.synsets)) == 999)
        print(f"Duplicating synset is 'crane'?".ljust(40), (imagenet.synsets[134] == imagenet.synsets[517]) & (imagenet.synsets[134] == ('crane',)))

        # Data attributes: check for split in ['train', 'val']
        print(f'Attributes `samples` is `imgs`?'.ljust(40), imagenet.samples is imagenet.imgs)
        print(f"Has correct number of samples?".ljust(40), len(imagenet.samples) == {'train':1281167, 'val':50000, 'test':100000}.get(imagenet.split))
        print(f"Has correct number of targets?".ljust(40), len(imagenet.targets) == {'train':1281167, 'val':50000, 'test':100000}.get(imagenet.split))

    smoke_test_imagenet()