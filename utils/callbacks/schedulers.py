from typing import Any, Callable, Optional, Sequence

from torch.optim.lr_scheduler import _LRScheduler

from .callback import Callback


class SchedulerOnEpoch(Callback):
    """Update an attribute of the host trainer at each epoch.

    Args:
    + `attribute`: Name of attribute.
    + `schedule_fn`: Schedule function that accepts the current `(epoch, value)`\
        and returns a new value.
    + `on_epoch`: Flag of which checkpoint of epoch to update the value:        \
        `'begin'`, `'train_end'`. Defaults to `'train_end'`.
    + `log`: Toggle to add the latest value to the trainer's history. Defaults  \
        to `True`.
    + `log_key`: Key to log in the trainer's history. Defaults to `'None'`, skip\
        to use `attribute`.
    
    Example: A schedule function that divides the value by 10 at epoch 40 & 80.
    ```
    def schedule_fn(epoch:int, value:float) -> float:
        if epoch in [40, 80]:
            return 0.1*value
    ```
    """
    def __init__(self,
        attribute:str,
        schedule_fn:Callable[[int, Any], Any],
        on_epoch:str='train_end',
        log:bool=True,
        log_key:Optional[str]=None,
    ):
        assert on_epoch in ['begin', 'train_end']
        super().__init__()
        self.attribute = attribute
        self.schedule_fn = schedule_fn
        self.on_epoch = on_epoch
        self.log = log
        self.log_key = log_key
        
        if self.log & (self.log_key is None):
            self.log_key = self.attribute

    def step(self, epoch:int):
        value = getattr(self.host, self.attribute)
        value = self.schedule_fn(epoch, value)
        setattr(self.host, self.attribute, value)

    def on_epoch_begin(self, epoch:int, logs:dict=None):
        if self.on_epoch == 'begin':
            self.step(epoch)
        if self.log is True:
            value = getattr(self.host, self.attribute)
            logs.update({self.log_key:value})

    def on_epoch_train_end(self, epoch:int, logs:dict=None):
        if self.on_epoch == 'train_end':
            self.step(epoch)


class ScheduleFunction:
    '''Base class for schedule functions.'''
    def __init__(self):
        self.params = {}
        
    def __call__(self, epoch:int, value:Any) -> Any:
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.params.items() if v is not None])})"


class StepScheduleFunction(ScheduleFunction):
    """Multiply the value by a constant at every step of a few iterations.
    Args:
    + `step_size`: Number of iterations of each step.
    + `gamma`: Multiplier at each step. Defaults to `0.1`.
    """
    def __init__(self, step_size:int, gamma:float|int=0.1):
        super().__init__()
        self.step_size = step_size
        self.gamma = gamma
        self.params.update({'step_size':step_size, 'gamma':gamma})
    
    def __call__(self, epoch:int, value:int|float) -> int|float:
        if (epoch > 0) & (epoch % self.step_size == 0):
            return self.gamma*value
        else:
            return value


class RepeatedScheduleFunction(ScheduleFunction):
    """Wraps around an repeats a schedule function.

    + `schedule_fn`: Schedule function to repeat.
    + `step_size`: Number of iterations of each repeat.
    + `value_init`: Initial value to reset to. Defaults to `None`, skip to
        automatically pick the first value it gets.
    """    
    def __init__(self, schedule_fn:ScheduleFunction, step_size:int, value_init:Optional[int|float]=None):
        super().__init__()
        self.schedule_fn = schedule_fn
        self.step_size = step_size
        self.value_init = value_init
        self.params.update({
            'schedule_fn': schedule_fn, 
            'step_size':   step_size,
            'value_init':  value_init,
        })
    
    def __call__(self, epoch:int, value:int|float) -> int|float:
        if (epoch == 0) & (self.value_init is None):
            self.value_init = value
            self.params.update({'value_init': self.value_init})
            
        if epoch % self.step_size == 0:
            value = self.value_init
        
        value = self.schedule_fn(epoch=epoch % self.step_size, value=value)
        return value


class LearningRateSchedulerOnEpoch(Callback):
    """Help a learning rate scheduler take a step at each epoch.

    Args:
    + `scheduler`: Scheduler to step.
    + `on_epoch`: Flag of which checkpoint of epoch to update the learning rate:\
        `'begin'`, `'train_end'`. Defaults to `'train_end'`.
    + `log`: Toggle to add the latest learning rate to the trainer's history.   \
        Defaults to `True`.
    + `log_key`: Key to log in the trainer's history. Defaults to `'lr'`.
    """
    def __init__(self,
        scheduler:_LRScheduler,
        on_epoch:str='train_end',
        log:bool=True,
        log_key:str='lr',
    ):
        assert on_epoch in ['begin', 'train_end']
        super().__init__()
        self.scheduler = scheduler
        self.on_epoch = on_epoch
        self.log = log
        self.log_key = log_key

    def on_epoch_begin(self, epoch:int, logs:dict=None):
        if self.on_epoch == 'begin':
            self.scheduler.step()
        
        if self.log is True:
            lrs:Sequence[float] = self.scheduler.get_last_lr()
            if len(lrs) == 1:
                logs.update({self.log_key:lrs[0]})
            else:
                logs.update({f'{self.log_key}_{idx_lr}':lr for idx_lr, lr in enumerate(lrs)})

    def on_epoch_train_end(self, epoch:int, logs:dict=None):
        if self.on_epoch == 'train_end':
            self.scheduler.step()