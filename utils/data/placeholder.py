import torch
from torch.utils.data import Dataset


class PlaceholderDataset(Dataset):
    """Dataset of only zeros to use as a placeholder in data-free settings.
        
    Args:
        `num_batches`: Number of batches. Defaults to `120`.
        `batch_size`: Batch size. Defaults to `512`.
    """
    def __init__(
        self,
        num_batches:int=120,
        batch_size:int=512,
    ):
        super().__init__()
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __len__(self):
        return self.num_batches*self.batch_size

    def __getitem__(self, index):
        return (torch.zeros([1]), torch.zeros([1]))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_batches={self.num_batches}, batch_size={self.batch_size})'
