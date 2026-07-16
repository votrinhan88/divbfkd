from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor

def show_aug(
    images:Tensor,
    augment_fn:Optional[Callable[[Any], Tensor]]=None,
    nrows:int=10,
    ncols:int=5,
    shuffle:bool=False,
    seed:Optional[int]=None,
    vmin=0,
    vmax=1,
    cmap='auto',
):
    num_examples = nrows*ncols
    if shuffle is True:
        images = images[torch.randperm(images.shape[0])]
    images = images[0:num_examples]
    
    fig, ax = plt.subplots(
        ncols=1 if augment_fn is None else 2,
        nrows=1,
        figsize=(ncols+0.5, nrows+0.5),
        constrained_layout=True,
        sharex='all',
        sharey='all',
        squeeze=False,
    )

    def tile_grid(images, ncols=ncols, nrows=nrows):
        # Tile images into a grid
        x = images.clone().detach()

        # Pad 1 pixel on top row & left column to all images in batch
        x = nn.functional.pad(x, pad=(1, 0, 1, 0), value=(vmin+vmax)/2) # top, bottom, left, right
        x = torch.reshape(x, shape=[nrows, ncols, *x.shape[1:]])
        x = torch.concat(torch.unbind(x, dim=0), dim=2)
        x = torch.concat(torch.unbind(x, dim=0), dim=2)
        # Crop 1 pixel on top row & left column from the concatenated image
        x = x[:, 1:, 1:]
        x = x.permute(1, 2, 0)
        x = x.clamp(min=vmin, max=vmax)
        return x

    if cmap == 'auto':
        cmap = 'gray' if images.shape[1] == 1 else None

    # Original images    
    images = tile_grid(images).squeeze(axis=-1)
    ax[0, 0].imshow(images, vmin=vmin, vmax=vmax, cmap=cmap)
    ax[0, 0].axis('off')
    ax[0, 0].set(title='Original')
    # Augmented images    
    if augment_fn is not None:
        if seed is None:
            augmented = augment_fn(images)
        else:
            cur_seed = torch.seed()
            torch.manual_seed(seed)
            augmented = augment_fn(images)
            torch.manual_seed(cur_seed)
        augmented = tile_grid(augmented).squeeze(axis=-1)
        ax[0, 1].imshow(augmented, vmin=vmin, vmax=vmax, cmap=cmap)
        ax[0, 1].set(title='Augmented')
    return fig, ax

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    from torchvision import transforms
    from kornia import augmentation
    from utils.data.fewshot import get_fewshot_dataset

    def smoke_test(
        dataset:str='MNIST',
        nrows:int=5,
        ncols:int=10,
        augment_fn:Optional[Callable[[Tensor], Tensor]]=None,
    ):
        fewshot_data = get_fewshot_dataset(dataset=dataset, num_examples=nrows*ncols)
        images = fewshot_data['x']
        fig, ax = show_aug(images=images, augment_fn=augment_fn, nrows=5, ncols=10)
        fig.show()

        print()