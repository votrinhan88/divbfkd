from typing import Optional
import numpy as np
from torch import Tensor
from PIL import Image

def view_array(
    array:np.ndarray|Tensor,
    preprocess:str='auto',
    path:str='./logs/temp.png',
    head:Optional[int]=5,
) -> Image:
    if isinstance(array, Tensor):
        array = array.cpu().numpy()
        if preprocess == 'auto':
            pass
    
    if array.dtype in [np.float32, np.float64]:
        array = (255*array).astype(np.uint8)
    
    # One black-and-white image: require exactly 2 channels for PIL.Image
    if array.ndim == 2:
        img = Image.fromarray(array)
        img.save(path)
        return img
    # One colored/black-and-white image: [H, W, C] | [C, H, W]?
    elif array.ndim == 3:
        # Likely [C, H, W]: change to [H, W, C]
        if array.shape[0] < array.shape[2]:
            array = array.transpose(1, 2, 0)
        # Squeeze black-and-white image
        if array.shape[2] == 1:
            array = array.squeeze(axis=2)
        img = Image.fromarray(array)
        img.save(path)
        return img
    # A batch of images: [N, H, W, C] | [N, C, H, W]?
    elif array.ndim == 4:
        # Likely [N, C, H, W]: change to [N, H, W, C]
        if array.shape[1] < array.shape[3]:
            array = array.transpose(0, 2, 3, 1)
        # A batch of black-and-white images:
        if array.shape[3] == 1:
            array = array.squeeze(axis=3)
        imgs = []
        for i in range(min(head, array.shape[0])):
            img = Image.fromarray(array[i])
            imgs.append(img)
            img.save(path.replace('.png', f'_{i:04d}.png'))
        return imgs