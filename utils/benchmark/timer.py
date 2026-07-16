from timeit import Timer
from typing import Callable, Optional, Sequence, Tuple

def time(
    task:Callable|str,
    setup:Callable|str='pass',
    repeat:int=1,
    number:Optional[int]=None,
) -> Sequence[Tuple[int, float]]|Sequence[float]:
    """Time a given task with timeit.
    
    Args:
        `task`: Task for timing.
        `setup`: Setup code which does not count towards timing. Defaults to   \
            `'pass'`.
        `repeat`: Number of runs timing is performed. Defaults to `1`.
        `number`: Number of times `task` is executed in each run. Defaults to  \
            `None`, skip to automatically increase from the sequence 1, 2, 5,  \
            10, 20, 50, … until the time taken is at least 0.2 second.
    
    Returns:
        A list of (number, time taken) if `number` is None.
        Or a list of time taken if `number` is an int.

    To pass in `task` with arguments, use `task = lambda:my_task(my_arg=...)`.
    """    
    timer = Timer(
        stmt=task,
        setup=setup,
    )

    if number is None:
        return [timer.autorange() for i in range(repeat)]
    else:
        return timer.repeat(repeat=repeat, number=number)

if __name__ == '__main__':
    import torch

    def task():
        x = torch.rand(size=[5])
        for i in range(1000):
            x += torch.rand(size=[5])
        return x
    
    print(f'{time(task=task)}')
    print(f'{time(task=lambda:task(), repeat=5, number=10)}')