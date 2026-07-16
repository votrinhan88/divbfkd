from typing import Any, Callable, Optional, Sequence

import numpy as np
import tqdm.auto as tqdm

from .callback import Callback


class ProgressBar(Callback):
    """Print a progress bar to screen. Included in trainers by default."""
    BLOCKS = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    
    def __init__(self):
        super().__init__()
        self.format_dict = {}
        self._format = self.format_default

    def register_format(self, format_dict:Optional[dict[str, Callable[[Any], str]]]=None):
        if self.format_dict is not None:
            self.format_dict.update(format_dict)
            if self.format_dict != dict():
                self._format = self.format_wrapper

    @staticmethod
    def format_default(k:str, v:float) -> str:
        return f'{v:.4g}'
    
    def format_multicounter(self, k:str, v:np.ndarray) -> str:
        max = v.max() + 1e-8
        norm_count = np.round(8*v/max).astype(int).tolist()
        return ''.join(['[', *[self.BLOCKS[i] for i in norm_count], ']'])
    
    def format_wrapper(self, k:str, v:float|Sequence[int|float]) -> str:
        return self.format_dict.get(k, self.format_default)(k=k, v=v)

    def on_train_begin(self, logs=None):
        self.kwargs = self.host.training_loop_kwargs
        start_epoch = self.kwargs['start_epoch']
        stop_epoch = self.kwargs['start_epoch'] + self.kwargs['num_epochs']
        self.host.training_progress = tqdm.tqdm(range(start_epoch, stop_epoch), desc='Training', unit='epochs')

    def on_test_begin(self, logs=None):
        self.kwargs = self.host.evaluate_kwargs
        self.host.val_phase_progress = tqdm.tqdm(self.kwargs['valloader'], desc='Evaluating', unit='batches')
    
    def on_epoch_train_end(self, epoch:int, logs=None):
        self.host.training_progress.set_postfix({k:self._format(k=k, v=v) for (k, v) in logs.items()})

    def on_epoch_end(self, epoch:int, logs=None):
        self.host.training_progress.set_postfix({k:self._format(k=k, v=v) for (k, v) in logs.items()})

    def on_epoch_begin(self, epoch:int, logs=None):
        self.host.train_phase_progress = tqdm.tqdm(self.kwargs['trainloader'], desc='Train phase', leave=False, unit='batches')

    def on_train_batch_end(self, batch:int, logs=None):
        self.host.train_phase_progress.set_postfix({k:self._format(k=k, v=v) for (k, v) in logs.items()})
        
    def on_epoch_test_begin(self, epoch:int, logs=None):
        self.host.val_phase_progress = tqdm.tqdm(self.kwargs['valloader'], desc='Test phase', leave=False, unit='batches')

    def on_test_batch_end(self, batch:int, logs=None):
        self.host.val_phase_progress.set_postfix({k:self._format(k=k, v=v) for (k, v) in logs.items()})