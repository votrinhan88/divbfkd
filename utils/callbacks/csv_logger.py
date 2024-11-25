import csv
import collections
import os

import numpy as np

from .callback import Callback


class CSVLogger(Callback):
    """Save all metrics in epoch log of trainer to a CSV file.

    Args:
    + `filename`: Path to save CSV to.
    + `separator`: Separator between elements of the same row. Defaults to     \
        `','`.
    + `append`: Flag to append to end or overwrite content of existing file.   \
        Defaults to `False`.
    """
    def __init__(self, filename:str, separator:str=',', append:bool=False):
        super().__init__()
        self.filename = filename
        self.separator = separator
        self.append = append
        
        self.writer = None
        self.keys = None
        self.append_header = True

    @staticmethod
    def handle_value(k):
        is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
        if isinstance(k, str):
            return k
        elif (
            isinstance(k, collections.abc.Iterable)
            and not is_zero_dim_ndarray
        ):
            return f"[{','.join(map(str, k))}]"
        else:
            return k

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.isfile(self.filename):
                with open(file=self.filename, mode="r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = open(file=self.filename, mode=mode, newline='')

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        if self.keys is None:
            train_keys, val_keys = [], []
            for key in logs.keys():
                if key[0:4] == 'val_':
                    val_keys.append(key)
                else:
                    train_keys.append(key)
            self.keys = train_keys + val_keys

        # if self.model.stop_training:
        #     # We set NA so that csv parsers do not fail for this last epoch.
        #     logs = dict(
        #         (k, logs[k]) if k in logs else (k, "NA") for k in self.keys
        #     )
        

        if self.writer is None:
            class CustomDialect(csv.excel):
                delimiter = self.separator
            
            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update(
            (key, self.handle_value(logs.get(key, float('nan')))) for key in self.keys
        )

        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None