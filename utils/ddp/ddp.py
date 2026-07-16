import os
import torch
import torch.distributed as dist

def setup_process(rank:int, world_size:int):
    # Use backend 'nccl' instead of 'gloo' for our server
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    if dist.get_backend() == 'nccl':
        torch.cuda.set_device(rank)

def setup_master():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(torch.randint(low=1025, high=65536, size=[1]).item())

def cleanup():
    dist.destroy_process_group()