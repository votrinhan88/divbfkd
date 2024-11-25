import cProfile
import pstats
from typing import Any, Callable, Optional


def profile(
    task:Callable[[Any], Any],
    filename:Optional[str]=None,
    print_stats:bool=True,
    *args, **kwargs,
):
    """Profile a task, can be viewed with snakeviz.
    
    Args:
    + `task`: Task for profiling. To pass in a function with arguments, use   \
        `task = lambda:my_task(my_arg=...)` or `functools.partial`.
    + `filename`: Path to save profiling data to, must ends with `'.prof'`.    \
        Defaults to `None`, skip to not save profiling data.
    + `print_stats`: Flag to print profiling data to terminal. Defaults to     \
        `True`.
    
    After the profiling is finished, view from bash with `snakeviz <filename>`.
    """
    with cProfile.Profile() as profiler:
        task(*args, **kwargs)

    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    
    if print_stats is True:
        stats.print_stats()
    if filename is not None:
        stats.dump_stats(filename=filename)


if __name__ == '__main__':
    import torch

    def task():
        x = torch.rand(size=[5])
        for i in range(1000):
            x += torch.rand(size=[5])
        return x
    
    profile(task)