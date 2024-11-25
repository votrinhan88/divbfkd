import glob
import os
from typing import Any, Callable, Optional

import shutil
from PIL import Image
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .callback import Callback
from utils.callbacks import Callback


class GIFMaker(Callback):
    """Base callback to generate a GIF image.

    Args:
    + `filename`: Path to save GIF to.
    + `postprocess_fn`: Post-processing function to map synthetic images back to\
        the plot range, ideally [0, 1]. Defaults to `None`, skip to skip        \
        post-processing.
    + `plt_norm`: Flag to use plt.imshow() automatic image normalization, else  \
        clamp to range [0, 1]. Defaults to `False`.
    + `seed`: Seed to ensure reproducibility between different runs. Defaults to\
        `None`.
    + `keep_last`: Flag to save last generated image. Defaults to `False`.
    + `delete_png`: Flag to delete PNG files and folder at `filename_png` after \
        training. Defaults to `True`.
    + `save_freq`: Number of epochs to produce an image. Defaults to `1`.
    + `duration`: Duration of the GIF in milliseconds. Defaults to `5000`.
    """
    def __init__(
        self,
        filename:str,
        postprocess_fn:Optional[Callable[[Any], Any]]=None,
        plt_norm:bool=False,
        seed:Optional[int]=None,
        keep_last:bool=False,
        delete_png:bool=True,
        save_freq:int=1,
        duration:float=5000,
    ):
        super().__init__()
        self.filename = filename
        self.postprocess_fn = postprocess_fn
        self.plt_norm = plt_norm
        self.seed = seed
        self.keep_last = keep_last
        self.delete_png = delete_png
        self.save_freq = save_freq
        self.duration = duration

        self.path_png_folder = self.filename[0:-4] + '_png'
        
        if self.postprocess_fn is None:
            self.postprocess_fn = lambda x:x
            
        if self.plt_norm is True:
            self.vmin, self.vmax = None, None
        elif self.plt_norm is False:
            self.vmin, self.vmax = 0, 1
    
    def on_train_begin(self, logs:dict = None):
        # Renew/create folder containing PNG files
        if os.path.isdir(self.path_png_folder):
            for png in glob.glob(f'{self.path_png_folder}/*.png'):
                os.remove(png)
        else:
            os.mkdir(self.path_png_folder)
    
    def make_gif(self):
        path_png = f'{self.path_png_folder}/*.png'
        sorted_path = sorted(glob.glob(path_png))

        # Keep last before making GIF and deleting all the images
        if (self.keep_last is True) & (len(sorted_path) >= 1):
            shutil.copy2(src=sorted_path[-1], dst=self.filename[0:-4]+'.png')
        
        # Load images
        img_objs = [Image.open(f) for f in sorted_path]

        # Make GIF if at least 2 images
        if len(sorted_path) >= 2:
            img, *imgs = img_objs
            img.save(
                fp=self.filename,
                format='GIF',
                append_images=imgs,
                save_all=True,
                duration=self.duration/(len(imgs) + 1),
                loop=0,
            )

        if self.delete_png is True:
            for png in glob.glob(path_png):
                os.remove(png)
            os.rmdir(self.path_png_folder)

    def modify_suptitle(self, figure:Figure, value:int):
        figure.suptitle(f'{self.host.__class__.__name__} - Epoch {value}')

    def modify_axis(self, axis:Axes):
        axis.axis('off')

    def modify_savepath(self, value:int) -> str:
        return f"{self.path_png_folder}/{self.host.__class__.__name__}_epoch_{value:04d}.png"