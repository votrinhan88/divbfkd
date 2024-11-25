from typing import Any, Callable, Literal, Optional, Tuple, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision

from .metadata import METADATA


def spread_dict(input:dict, groups:Optional[Sequence]=None) -> dict[Any, dict]:
    """Spread a dict of `{key: {group: value}}` to a dict of `{group: {key: value}}`.

    Args:
    + `input`: The dict to spread.
    + `groups`: First-level keys to spread items to. Defaults to `None` to find \
        all possible groups from `input`.
    
    Returns:
    + A dict spreaded by `groups`.
    """
    if groups is None:
        # Find all possible groups
        groups = []
        for v in input.values():
            if isinstance(v, dict):
                groups.extend(v.keys())
    
    # Spread values to separate groups
    spreaded = {g: {} for g in groups}
    for g in groups:
        for k, v in input.items():
            if isinstance(v, dict):
                spreaded[g][k] = v.get(g)
            else:
                spreaded[g][k] = v
    return spreaded


def spread_item(input, groups:Optional[Sequence]=None) -> dict:
    """Spread input items to a dict by groups.

    Args:
    + `input`: An item to spread, or dict to check for matching groups.
    + `groups`: First-level keys to spread items to or check if they are matched\
        with the keys in `input`. Defaults to `None` to skip checking.
    
    Returns:
    + A dict of `input` spreaded by `groups`.
    """
    if groups is None:
        if isinstance(input, dict):
            return input
        else:
            raise ValueError("Cannot spread a single `input` without specifying `groups`.")
    else:
        if isinstance(input, dict):
            for g in groups:
                assert g in input.keys(), (
                    f"Value for group '{g}' is missing.")
        else:
            spreaded = {g:input for g in groups}
    return spreaded


def get_transform(
    metadata:Optional[dict]=None,
    init_augmentation:Optional[Callable]=None,
    resize:Optional[Tuple[float, float]]=None,
    rescale:Optional[Literal['standardization']| Tuple[float, float]]=None,
):
    transform = []

    if init_augmentation is not None:
        # Unpack transforms in `torchvision.transforms.Compose` instance
        if isinstance(init_augmentation, torchvision.transforms.Compose):
            transform.extend(init_augmentation.transforms)
        elif isinstance(init_augmentation, Sequence):
            transform.extend(init_augmentation)
        else:
            transform.append(init_augmentation)

    # Resize
    if resize is not None:
        transform.append(torchvision.transforms.Resize(size=resize))

    # Convert to tensors
    transform.append(torchvision.transforms.ToTensor())

    # Rescaling (Normalization)
    if rescale is not None:
        if rescale == 'standardization':
            mean, std = metadata.get('mean_std')
            mean = torch.tensor(mean)
            std = torch.tensor(std)
        else:
            mean = -rescale[0]/(rescale[1] - rescale[0])
            std = 1/(rescale[1] - rescale[0])
        transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
    
    # Compose the transforms
    if len(transform) > 1:
        return torchvision.transforms.Compose(transform)
    else:
        return transform[0]


def get_dataset(
    name:str,
    root:Optional[str]=None,
    splits:Optional[str]=None,
    init_augmentations:Optional[Callable|dict[str, Callable]]=None,
    resize:Optional[Tuple[float, float]]=None,
    rescale:Optional[Literal['standardization']| Tuple[float, float]]=None,
    onehot_label:bool=False,
    **kwargs,
) -> dict[str, Dataset]:
    """Get dataset from prepared pipeline. See utils.data.metadata for currently
    supported datasets.

    Args: Except for `name` and `root`, the arguments should be in the format of\
    a dict of `(split, value)`.
    + `name`: Dataset name.
    + `root`: Dataset directory. Defaults to `None`, skip to use                \
        './datasets/<name>/'.
    + `splits`: Splits of the dataset. Defaults to `None`, skip to use the      \
        defaults in the metadata.
    + `init_augmentations`: Augmentation transforms for the inputs as a dict of \
        `(split: transform)`. Defaults to `None`, skip to skip augmentation.
    + `resize`: A tuple of `(width, height)` to resize examples to. Defaults to \
        `None`, skip to skip resizing.
    + `rescale`: A tuple of `(min, max)` bounds to rescale examples to, or pass \
        in `'standardization'` to rescale to mean = 0, std = 1 with precomputed \
        values. Defaults to `None`, skip to skip rescaling.
    + `onehot_label`: Flag for one-hot encoded labels. Defaults to `False`.

    Kwargs:
    + Additional arguments to pass into the `Dataset` class.

    Returns:
    + A dict of `(split: dataset)`.
    """
    if name not in METADATA.keys():
        raise NotImplementedError(f"Dataset '{name}' is not implemented.")
    metadata = METADATA.get(name)
    
    if root == None:
        root = f'./datasets/{name}/'
    
    if splits == None:
        splits = metadata.get('splits')
    else:
        for s in splits:
            if s not in metadata.get('splits'):
                raise ValueError(f"Split '{s}' is not available for dataset '{name}'.")

    def onehot(y):
        return torch.zeros(metadata.get('num_classes')).scatter_(0, torch.tensor(y), value=1)

    # Prepare `transforms` and `target_transforms`
    if not isinstance(init_augmentations, dict):
        init_augmentations = {s:init_augmentations for s in splits}
    else:
        for s in splits:
            assert s in init_augmentations.keys(), (
                f"`init_augmentations` for split '{s}' is missing.")
    
    transforms = {s:[] for s in splits}
    target_transforms = {s:None for s in splits}
    for s in splits:
        transforms[s] = get_transform(
            metadata=metadata,
            init_augmentation=init_augmentations.get(s),
            resize=resize,
            rescale=rescale,
        )
        target_transforms[s] = onehot if onehot_label else None

    dataset = {s:None for s in splits}
    for s in splits:
        if name in ['MNIST', 'FashionMNIST', 'USPS', 'CIFAR10', 'CIFAR100']:
            dataset[s] = metadata.get('DatasetClass')(
                root=root,
                train={'train':True, 'test':False}.get(s),
                transform=transforms.get(s),
                target_transform=target_transforms.get(s),
                **kwargs,
            )
        else:
            dataset[s] = metadata.get('DatasetClass')(
                root=root,
                split=s,
                transform=transforms.get(s),
                target_transform=target_transforms.get(s),
                **kwargs,
            )
    return dataset


def get_dataloader(
    dataset:dict[str, Dataset],
    ddp:bool=False,
    ddp_kwargs:dict[str, Any]=None,
    **kwargs,
) -> dict[str, DataLoader]:
    """Get dataloader for the given dataset in the pipeline.

    Args:
    + `dataset`: Dataset in the format of a dict of `(split: dataset)`.
    + `ddp`: Flag to enable support for Distributed Data Parallel training.     \
        Defaults to `False`.
    + `ddp_kwargs`: A dict of additional arguments to pass into the             \
        `DistributedSampler` class, including: rank, world_size, shuffle, seed, \
        drop_last. The arguments can be of a single value or a dict of `(split, \
        value)`. Defaults to `None`.

    Kwargs:
    + Additional arguments to pass into the `DataLoader` class.

    Returns:
    + A dict of `(split: dataloader)`.
    """
    # Parse DDP config
    if ddp:
        if ddp_kwargs == None:
            ddp_kwargs = {}
        for k in ['world_size', 'rank']:
            if ddp_kwargs.get(k) == None:
                raise ValueError(f'`ddp_kwargs` needs to include `{k}`.')
        ddp_kwargs['num_replicas'] = ddp_kwargs.pop('world_size')

    dataloader = {}
    for split in dataset.keys():
        split_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, dict):
                split_kwargs[k] = v[split]
            else:
                split_kwargs[k] = v
        
        # Support for Distributed Data Parallel training
        if ddp:
            split_ddp_kwargs = {}
            for k, v in ddp_kwargs.items():
                if isinstance(v, dict):
                    split_ddp_kwargs[k] = v[split]
                else:
                    split_ddp_kwargs[k] = v
            
            # For stability, it is recommended to set:
            #   + num_workers = 0 (extra threads in the children processes may be problemistic)
            #   + pin_memory = False
            split_kwargs['pin_memory'] = split_kwargs.get('pin_memory', False)
            split_kwargs['num_workers'] = split_kwargs.get('num_workers', 0)
                    
            sampler = DistributedSampler(
                dataset=dataset.get(split),
                **split_ddp_kwargs,
            )
            dataloader[split] = DataLoader(
                dataset=dataset.get(split),
                sampler=sampler,
                **split_kwargs,
            )
        else:
            dataloader[split] = DataLoader(
                dataset=dataset.get(split),
                **split_kwargs,
            )
    
    return dataloader


def compute_mean_std(dataset, **kwargs):
    """Compute the normalization coefficients (mean and standard deviation) for 
    a given image dataset - to have mean = 0 and std = 1 in each channel.
    Images are assumed to have the same dimension and the number of channels.

    Args:
    + `dataset`: Dataset to compute mean and standard deviation.
    
    Kwargs: Additional arguments to `torch.utils.data.DataLoader`.
    
    Returns:
    + A tuple of `(mean, std)`.
    """        
    dataloader = DataLoader(dataset=dataset, **kwargs)
    
    sum_mean = torch.zeros(size=[1], dtype=torch.float64)
    for images, _ in dataloader:
        images:Tensor = images.view(*images.shape[0:2], -1)
        sum_mean = sum_mean + images.mean(dim=2).sum(dim=0)
    mean = sum_mean/len(dataset)
        
    sum_var_per_pixel = torch.zeros(size=[1], dtype=torch.float64)
    for images, _ in dataloader:
        images:Tensor = images.view(*images.shape[0:2], -1)
        sum_var_per_pixel = sum_var_per_pixel + ((images - mean.unsqueeze(dim=1))**2).mean(dim=2).sum(dim=0)
    std = (sum_var_per_pixel/len(dataset)).sqrt()
    return mean, std