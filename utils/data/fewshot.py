import copy
import random
from typing import Optional, Tuple
import warnings

from numpy import ndarray
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .metadata import METADATA


def set_difference(x1:Tensor, x2:Tensor) -> Tensor:
    """Compute the set difference of two set tensors, each containing unique
    values.

    Args:
    + `x1`: The first set tensor.
    + `x2`: The second set tensor.

    Returns:
    + The set difference `x1 \ x2`
    """    
    combined = torch.cat((x1, x2, x2))
    uniques, counts = combined.unique(return_counts=True)
    x1_set_diff_x2 = uniques[counts == 1]
    return x1_set_diff_x2


def make_fewshot(
    dataset:Dataset,
    num_samples:int=100,
    balanced:bool=True,
    shuffle:bool=False,
    attr_inputs:Optional[str]=None,
    attr_targets:Optional[str]=None,
    seed:Optional[int]=None,
    with_index:bool=False,
) -> Dataset|Tuple[Dataset, Tensor]:
    """A wrapper function to monkey-patch a `torch.utils.data.Dataset` instance 
    into a few-shot dataset.

    Args:
    + `dataset`: A dataset instance.
    + `num_samples`: Total number of samples to select. Defaults to `100`.
    + `balanced`: Flag to make the dataset have balanced classes. However, in   \
        case any class does not have enough samples, random samples of other    \
        classes will be selected. Defaults to `True`.
    + `shuffle`: Flag to shuffle the dataset before returning. Defaults to      \
        `True`.
    + `attr_inputs`: Name of attribute of the inputs. If not specify, will      \
        attempt to search from metadata then guess from `['data', 'samples']`.  \
        Defaults to `None`.
    + `attr_targets`: Name of attribute of the targets. If not specify, will    \
        attempt to search from metadata then guess from `['labels', 'targets']`.\
        Defaults to `None`.
    + `seed`: Seed for reproducibility. Defaults to `None`.
    + `with_index`: Flag to return selected indices along with the fewshot      \
        dataset. Defaults to `False`.
    
    Returns:
    + The fewshot `dataset` instance, or the tuple `(dataset, indices)` if      \
        `with_index` is True.
    """
    few_set:Dataset = copy.copy(dataset)
    dataset_cls = few_set.__class__.__name__
    if seed is not None:
        torch.manual_seed(seed=seed)

    if attr_inputs is None:
        if dataset_cls in METADATA.keys():
            attr_inputs = METADATA.get(dataset_cls).get('attr_inputs')
        else:
            for attr in ['data', 'samples']:
                if hasattr(few_set, attr):
                    attr_inputs = attr
        if attr_inputs is None:
            raise ValueError(
                f"Cannot find `attr_inputs` for dataset '{dataset_cls}' ." \
                f"Please specify explicitly."
            )
    if attr_targets is None:
        if dataset_cls in METADATA.keys():
            attr_targets = METADATA.get(dataset_cls).get('attr_targets')
        else:
            for attr in ['labels', 'targets']:
                if hasattr(few_set, attr):
                    attr_targets = attr
        if attr_targets is None:
            raise ValueError(
                f"Cannot find `attr_targets` for dataset '{dataset_cls}' ." \
                f"Please specify explicitly."
            )
    if num_samples > len(few_set):
        raise ValueError(
            f'The dataset {dataset_cls} has less than {num_samples} samples. ' 
            f'Please specify a smaller `num_samples`.'
        )
    # Determine classes
    targets = getattr(few_set, attr_targets)
    if isinstance(targets, Tensor):
        targets = targets.clone()
    else:
        targets = torch.tensor(targets)
    classes:Tensor = targets.unique()

    if balanced == True:
        # Group sample indices by class
        remain_each_class = [[] for k in classes]
        for idx, clss in enumerate(targets):
            remain_each_class[clss].append(idx)
        remain_hist = torch.tensor([len(indexes) for indexes in remain_each_class])
        taken = [[] for k in classes]
        taken_hist = torch.tensor([len(indexes) for indexes in taken])

        # Init variables for while-looping
        remain_classes = [k for k in classes]
        num_remain_samples = num_samples

        while num_remain_samples > 0:
            if num_remain_samples >= len(remain_classes):
                greedy_take = num_remain_samples // len(remain_classes)

                for k in torch.tensor(remain_classes):
                    num_to_take = min(greedy_take, remain_hist[k])
                    # Take random samples
                    shuffled = torch.tensor(remain_each_class[k])
                    shuffled = shuffled[torch.randperm(shuffled.shape[0])]
                    idx_to_take = shuffled[0:num_to_take]
                    
                    taken[k].extend(idx_to_take.tolist())
                    remain_each_class[k] = set_difference(torch.tensor(remain_each_class[k]), idx_to_take).tolist()

                    if len(remain_each_class[k]) == 0:
                        remain_classes.remove(k)
            
            else:
                # Needed samples < number of remaining classes
                # Take one sample from of the remaining classes
                random.shuffle(remain_classes)
                random_remain_classes = remain_classes[0:num_remain_samples]
                for k in random_remain_classes:
                    shuffled = torch.tensor(remain_each_class[k])
                    shuffled = shuffled[torch.randperm(shuffled.shape[0])]
                    idx_to_take = shuffled[0].item()
                    
                    taken[k].append(idx_to_take)
                    remain_each_class[k].remove(idx_to_take)

            remain_hist = torch.tensor([len(indexes) for indexes in remain_each_class])
            taken_hist = torch.tensor([len(indexes) for indexes in taken])
            num_remain_samples = num_samples - taken_hist.sum(dim=0)

        if taken_hist.max() != taken_hist.min():
            warnings.warn(
                f'Extracted fewshot dataset {dataset_cls} is imbalanced, with'  \
                f' {taken_hist.min()}-{taken_hist.max()} samples per class.'
            )
        
        # Aggregate few-shot indices
        taken_agg = torch.zeros(size=[0], dtype=torch.long)
        for k in classes:
            taken_agg = torch.cat((taken_agg, torch.tensor(taken[k], dtype=torch.long)), dim=0)
    
    if shuffle:
        taken_agg = taken_agg[torch.randperm(taken_agg.shape[0])]

    # Monkey-patch dataset to few-shot version
    inputs = getattr(few_set, attr_inputs)
    if isinstance(inputs, (Tensor, ndarray)):
        inputs = inputs[taken_agg]
    else:
        inputs = [inputs[i] for i in taken_agg.tolist()]
    setattr(few_set, attr_inputs, inputs)
    setattr(few_set, attr_targets, targets[taken_agg])

    if with_index == True:
        return few_set, taken_agg
    else:
        return few_set