from typing import Optional

import torch
from torch import Tensor
import torch.distributed as dist


def all_gather_nd(tensor:Tensor, world_size:Optional[int]=None) -> list[Tensor]:
    """Gathers tensor arrays of different lengths in a list.

    The length dimension is 0. This supports any number of extra dimensions in
    the tensors. All the other dimensions should be equal between the tensors.

    Args:
    + `tensor`: Tensor to be broadcast from current process.
    + `world_size`: The number of processes in case of Distributed Data Parallel\
        training. Defaults to `None`, skips to query automatically with `torch`.

    Returns:
    + List of gathered tensors of different sizes.
    
    Source: https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
    """
    if world_size is None:
        world_size = dist.get_world_size()
    local_size = torch.tensor(tensor.size(), device=tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_length = max(size[0] for size in all_sizes)
    print(f'all_sizes[0].tolist() {all_sizes[0].tolist()}')
    dist.barrier()
    if max_length == 0:
        return [torch.zeros(size=all_sizes[0].tolist())]*world_size

    length_diff = max_length.item() - local_size[0].item()
    if length_diff:
        pad_size = (length_diff, *tensor.size()[1:])
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding))

    all_tensors_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(all_tensors_padded, tensor)
    all_tensors = []
    for tensor_, size in zip(all_tensors_padded, all_sizes):
        print(tensor_, size)
        all_tensors.append(tensor_[:size[0]])
    return all_tensors