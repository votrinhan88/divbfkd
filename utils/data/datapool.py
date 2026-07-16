from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class TensorPool(Dataset):
    """A data pool with fields of type Tensor.

    Args:
    + `dims`: Dimension of tensors, skipping the first (batch) dimension, nested\
        in a list.
    + `dtypes`: Dtypes of tensors, single-value or nested in a list. Defaults to\
        `None`, skip to use `torch.float`.
    + `transforms`: Transform functions, single-value or nested in a list.      \
        Defaults to `None`, skip to skip transform.
    + `device`: Device to store the pool. Defaults to `'cpu'`.
    """
    def __init__(self,
        tensors:Optional[Sequence[Tensor]]=None,
        dims:Sequence[None|int|Sequence[int]]=None,
        dtypes:Optional[torch.dtype|Sequence[torch.dtype]]=None,
        transforms:Optional[Callable[[Tensor], Tensor]|Sequence[Callable[[Tensor], Tensor]]]=None,
        device:str='cpu',
    ):
        if (tensors is None) == (dims is None):
            raise ValueError('Please specify only either `tensors` or `dims`.')
        
        self.transforms:list = transforms
        self.device = torch.device(device)
        
        # Parse by tensors
        if tensors != None:
            assert isinstance(tensors, Sequence), '`tensors` must be a list of Tensors.'
            for t in tensors:
                assert isinstance(t, Tensor), '`tensors` must be a list of Tensors.'
            
            self.num_fields = len(tensors)
            self.fields = [t.clone().detach().to(device=self.device) for t in tensors]
            self.dims = [list(f.shape[1:]) for f in self.fields]
            self.dims = [None if d in [[], ()] else d for d in self.dims]
            self.dtypes = [f.dtype for f in self.fields]
        # Parse by dims
        elif dims != None:
            self.num_fields = len(dims)
            self.dims:list   = dims
            self.dtypes:list = dtypes

            # Parse dtypes
            if self.dtypes is None:
                self.dtypes = [torch.float]*self.num_fields
            elif isinstance(self.dtypes, torch.dtype):
                self.dtypes = [self.dtypes]*self.num_fields
            else:
                assert len(self.dtypes) == self.num_fields, (
                    f'Numbers of specified `dims` ({self.num_fields}) and `dtypes`    \
                    ({len(self.dtypes)}) are different.'
                )
            
            # Pre-allocate fields
            self.fields:list[Tensor] = [None]*self.num_fields
            for i, (dim, dtype) in enumerate(zip(self.dims, self.dtypes)):
                if dim is None:
                    self.fields[i] = torch.empty(size=[0], dtype=dtype, device=self.device)
                elif isinstance(dim, int):
                    self.fields[i] = torch.empty(size=[0, dim], dtype=dtype, device=self.device)
                else:
                    self.fields[i] = torch.empty(size=[0, *dim], dtype=dtype, device=self.device)

        # Parse transforms
        if self.transforms is None:
            self.transforms = [None]*self.num_fields
        elif isinstance(self.transforms, Callable):
            self.transforms = [self.transforms]*self.num_fields
        else:
            assert len(self.transforms) == self.num_fields, (
                f'Numbers of specified `dims` ({self.num_fields}) and `transforms`\
                ({len(self.transforms)}) are different.'
            )

    def __getitem__(self, index) -> Sequence[Tensor]:
        return [t(f[index]) if t is not None else f[index]
                    for f, t in zip(self.fields, self.transforms)]

    def slice(self, indices:Sequence[int]|Tensor):
        if isinstance(indices, Sequence):
            indices = torch.tensor(indices)
        
        for i in range(self.num_fields):
            self.fields[i] = self.fields[i][indices]

    def add(self, new_data:Sequence[Tensor]):
        if len(new_data) != self.num_fields:
            raise ValueError(
                f'Mismatched number of new data objects ({len(new_data)}) and ' \
                f'number of fields ({self.num_fields}).'
            )
        
        for i, datum in enumerate(new_data):
            datum = datum.detach().clone().to(device=self.device)
            self.fields[i] = torch.cat(tensors=[self.fields[i], datum], dim=0)
    
    def delete(self, indices:Optional[Sequence[int]|Tensor]=None):
        if indices is None:
            filtered_id = torch.empty(size=[0])
        else:
            if isinstance(indices, Tensor):
                indices = indices.tolist()
            indices = set(indices)

            all_id = set(torch.arange(len(self)).tolist())
            filtered_id = torch.tensor(list(all_id - indices))

        for i in range(self.num_fields):
            if filtered_id.numel() > 0:
                self.fields[i] = self.fields[i][filtered_id]
            elif filtered_id.numel() == 0:
                self.fields[i] = torch.empty(
                    size=[0, *self.fields[i].shape[1:]],
                    device=self.device,
                    dtype=self.fields[i].dtype,
                )

    def overwrite(self, indices:Sequence[int]|Tensor, new_data:Sequence[Tensor]):
        if len(new_data) != self.num_fields:
            raise ValueError(
                f'Mismatched number of new data objects ({len(new_data)}) and ' \
                f'number of fields ({self.num_fields}).'
            )

        if isinstance(indices, Sequence):
            overwrite_len = len(indices)
        elif isinstance(indices, Tensor):
            overwrite_len = indices.shape[0]
        for i, datum in enumerate(new_data):
            if overwrite_len != datum.shape[0]:
                raise ValueError(
                    f'Mismatched shape for indices ({overwrite_len}) and data ' \
                    f'at position {i} ({datum.shape[0]}).'
                )
            
        for i, datum in enumerate(new_data):
            datum = datum.detach().clone().to(device=self.device)
            self.fields[i][indices] = datum

    def shuffle(self, indices:Optional[Sequence[int]|Tensor]=None):
        if indices is None:
            shuffled = torch.randperm(len(self))
        else:
            if isinstance(indices, Sequence):
                indices = torch.tensor(indices)
            shuffled = indices[torch.randperm(indices.shape[0])]

        for i in range(self.num_fields):
            self.fields[i][indices] = self.fields[i][shuffled]

    def add_field(self,
        field:Tensor,
        transform:Optional[Callable[[Any], Any]]=None,
    ):
        if field.shape[0] != len(self):
            raise ValueError(
                f'The new field has mismatching length (TensorPool: {len(self)}'
                f', field: {field.shape[0]}).'
            )
        
        self.fields.append(field.clone().detach().to(device=self.device))
        self.num_fields += 1
        dim = field.shape[1:]
        self.dims.append(None if dim in [[], ()] else dim)
        self.dtypes.append(field.dtype)
        self.transforms.append(transform)

    def get_loader(self, **kwargs):
        return DataLoader(dataset=self, **kwargs)

    def __len__(self) -> int:
        return self.fields[0].shape[0]

    def __repr__(self):
        args = {
            'dims':self.dims,
            'dtypes':self.dtypes,
            'transforms':self.transforms,
            'device':self.device,
        }
        args = ', '.join([f'{k}={v}' for k, v in args.items() if v is not None])
        return f'{self.__class__.__name__}({args})'


class NdArrayPool(Dataset):
    """A data pool with fields of type ndarray to store arbitrary dtypes. More
    flexible but needs more catering.

    Args:
    + `getitems`: Getitem functions, single-value or nested in a list.      \
        Defaults to `None`, skip to skip transform.
    + `transforms`: Transform functions, single-value or nested in a list.      \
        Defaults to `None`, skip to skip transform.
    """
    def __init__(self,
        fields:Optional[Sequence[np.ndarray|Any]]=None,
        dims:Optional[Sequence[int|Sequence[int]|None]]=None,
        dtypes:Optional[np.dtype|Sequence[np.dtype]]=None,
        getitems:Optional[Callable|Sequence[Callable]]=None,
        transforms:Optional[Callable|Sequence[Callable]]=None,
    ):
        if (fields is None) == (dims is None):
            raise ValueError('Please specify only either `fields` or `dims`.')

        if fields != None:
            assert isinstance(fields, Sequence), '`fields` must be a list of data.'

            self.num_fields = len(fields)
            self.fields = [self.make_numpy_clone(f) for f in fields]
            for i in np.arange(start=1, stop=self.num_fields, step=1):
                if self.fields[0].shape[0] != self.fields[i].shape[0]:
                    raise ValueError(
                        f'The field at position {i} has mismatching length '
                        f'(Length: {[f.shape[0] for f in self.fields]}).'
                    )
            self.dims = [f.shape[1:] for f in self.fields]
            self.dims = [None if d in [[], ()] else d for d in self.dims]
            self.dtypes = [f.dtype for f in self.fields]
        elif dims != None:
            self.num_fields = len(dims)
            self.dims   = dims
            self.dtypes = dtypes

            # Parse dtypes
            if self.dtypes is None:
                self.dtypes = ['O']*self.num_fields
            elif isinstance(self.dtypes, np.dtype):
                self.dtypes = [self.dtypes]*self.num_fields
            else:
                assert len(self.dtypes) == self.num_fields, (
                    f'Numbers of specified `dims` ({self.num_fields}) and `dtypes`    \
                    ({len(self.dtypes)}) are different.'
                )

            # Pre-allocate fields
            self.fields = [None]*self.num_fields
            for i, (dim, dtype) in enumerate(zip(self.dims, self.dtypes)):
                if dim is None:
                    self.fields[i] = np.empty(shape=[0], dtype=dtype)
                elif isinstance(dim, int):
                    self.fields[i] = np.empty(shape=[0, dim], dtype=dtype)
                else:
                    self.fields[i] = np.empty(shape=[0, *dim], dtype=dtype)

        self.getitems:list = getitems
        self.transforms:list = transforms
        self.num_fields = len(self.dims)

        # Parse dtypes
        if self.dtypes is None:
            self.dtypes = [np.dtype('O')]*self.num_fields
        elif isinstance(self.dtypes, np.dtype):
            self.dtypes = [self.dtypes]*self.num_fields
        else:
            assert len(self.dtypes) == self.num_fields, (
                f'Numbers of specified `dims` ({self.num_fields}) and `dtypes`    \
                ({len(self.dtypes)}) are different.'
            )

        # Parse getitems
        if self.getitems is None:
            self.getitems = [None]*self.num_fields
        elif isinstance(self.getitems, Callable):
            self.getitems = [self.getitems]*self.num_fields
        else:
            assert len(self.getitems) == self.num_fields, (
                f'Numbers of specified `dims` ({self.num_fields}) and `getitems`\
                ({len(self.getitems)}) are different.'
            )

        # Parse transforms
        if self.transforms is None:
            self.transforms = [None]*self.num_fields
        elif isinstance(self.transforms, Callable):
            self.transforms = [self.transforms]*self.num_fields
        else:
            assert len(self.transforms) == self.num_fields, (
                f'Numbers of specified `dims` ({self.num_fields}) and `transforms`\
                ({len(self.transforms)}) are different.'
            )

    def __getitem__(self, index:int) -> Sequence:
        data = [None]*self.num_fields
        for i, (f, g, t) in enumerate(zip(self.fields, self.getitems, self.transforms)):
            datum = f[index]
            if g is not None:
                datum = g(datum)
            if t is not None:
                datum = t(datum)
            data[i] = datum
        return data

    def slice(self, indices:Sequence[int]|np.ndarray):
        indices = self.make_numpy_clone(indices)
        for i in range(self.num_fields):
            self.fields[i] = self.fields[i][indices]

    def add(self, new_data:Sequence[np.ndarray|Any]):
        if len(new_data) != self.num_fields:
            raise ValueError(
                f'Mismatched number of new data objects ({len(new_data)}) and ' \
                f'number of fields ({self.num_fields}).'
            )
        
        for i, datum in enumerate(new_data):
            datum = self.make_numpy_clone(datum)
            self.fields[i] = np.concatenate([self.fields[i], datum], axis=0)
    
    def delete(self, indices:Optional[Sequence[int]]=None):
        if indices is None:
            indices = np.arange(len(self))

        for i in range(self.num_fields):
            self.fields[i] = np.delete(arr=self.fields[i], obj=indices, axis=0)

    def overwrite(self, indices:Sequence[int]|Tensor, new_data:Sequence[np.ndarray|Any]):
        if len(new_data) != self.num_fields:
            raise ValueError(
                f'Mismatched number of new data objects ({len(new_data)}) and ' \
                f'number of fields ({self.num_fields}).'
            )

        if isinstance(indices, Sequence):
            overwrite_len = len(indices)
        elif isinstance(indices, (np.ndarray, Tensor)):
            overwrite_len = indices.shape[0]
        for i, datum in enumerate(new_data):
            datum_len = self.make_numpy_clone(datum).shape[0]
            if overwrite_len != datum_len:
                raise ValueError(
                    f'Mismatched shape for indices ({overwrite_len}) and data ' \
                    f'at position {i} ({datum_len}).'
                )
            
        for i, datum in enumerate(new_data):
            datum = self.make_numpy_clone(datum)
            self.fields[i][indices] = datum

    def shuffle(self, indices:Optional[Sequence[int]]=None):
        if indices is None:
            shuffled = np.random.permutation(len(self))
        else:
            if isinstance(indices, Sequence):
                indices = np.array(indices)
            shuffled = indices[np.random.permutation(indices.shape[0])]

        for i in range(self.num_fields):
            self.fields[i][indices] = self.fields[i][shuffled]

    @staticmethod
    def make_numpy_clone(obj) -> np.ndarray:
        if isinstance(obj, np.ndarray):
            return obj.copy()
        elif isinstance(obj, Tensor):
            return obj.clone().detach().to(device='cpu').numpy()
        else:
            return np.array(obj)

    def add_field(self,
        field:np.ndarray|Any,
        getitem:Optional[Callable]=None,
        transform:Optional[Callable]=None,
    ):
        field = self.make_numpy_clone(field)
        if field.shape[0] != len(self):
            raise ValueError(
                f'The new field has mismatching length (NdArrayPool: {len(self)}'
                f', field: {field.shape[0]}).'
            )
        
        self.fields.append(field)
        self.num_fields += 1
        dim = field.shape[1:]
        self.dims.append(None if dim in [[], ()] else dim)
        self.dtypes.append(field.dtype)
        self.getitems.append(getitem)
        self.transforms.append(transform)
    
    def get_loader(self, **kwargs):
        return DataLoader(dataset=self, **kwargs)

    def __len__(self) -> int:
        return self.fields[0].shape[0]

    def __repr__(self):
        args = {
            'dims':self.dims,
            'dtypes':self.dtypes,
            'getitems':self.getitems,
            'transforms':self.transforms,
        }
        args = ', '.join([f'{k}={v}' for k, v in args.items() if v is not None])
        return f'{self.__class__.__name__}({args})'


if __name__ == '__main__':
    def test_TensorPool():
        print(' Test for TensorPool: '.center(80,'#'))

        a = torch.rand(5, 3, 2)
        b = torch.rand(5, 1)
        c = torch.rand(5)
        tensor_pool = TensorPool(tensors = [a, b, c])

        flag_init = ((tensor_pool.fields[0] == a).all() & (not (tensor_pool.fields[0] is a))
            & (tensor_pool.fields[1] == b).all() & (not (tensor_pool.fields[1] is b))
            & (tensor_pool.fields[2] == c).all() & (not (tensor_pool.fields[2] is c))
        )
        print(f'__init__: {flag_init}.')

        a_0, b_0, c_0 = tensor_pool[0]
        flag_getitem = ((a_0 == a[0]).all()) & ((b_0 == b[0]).all()) & ((c_0 == c[0]).all())
        print(f'__getitem__(): {flag_getitem}.')

        a_extra = torch.ones(size=(1, 3, 2))
        b_extra = torch.ones(size=(1, 1))
        c_extra = torch.ones(size=(1,))
        tensor_pool.add([a_extra, b_extra, c_extra])
        a_last, b_last, c_last = tensor_pool[-1]
        flag_add = (len(tensor_pool) == 6) & (
            ((not (a_last is a_extra)) & (a_last == a_extra).all())
            & ((not (b_last is b_extra)) & (b_last == b_extra).all())
            & ((not (c_last is c_extra)) & (c_last == c_extra).all())
        )
        print(f'add(): {flag_add}.')

        tensor_pool.delete(indices=[0, 5])
        flag_delete = (len(tensor_pool) == 4) & (
            (tensor_pool.fields[0] == a[1:]).all()
            & (tensor_pool.fields[1] == b[1:]).all()
            & (tensor_pool.fields[2] == c[1:]).all()
        )
        print(f'delete(): {flag_delete}.')

        a_temp = tensor_pool.fields[0].clone()
        tensor_pool.shuffle(indices=[0, 1, 2])
        flag_shuffle = (
            (tensor_pool.fields[0][0:3] != a_temp[0:3]).any()
            & (tensor_pool.fields[0][3] == a_temp[3]).all()
        )
        print(f'shuffle(): {flag_shuffle} - May need multiple runs.')

        d = torch.rand(size=[4, 2])
        tensor_pool.add_field(field=d)
        flag_add_field = (
            (tensor_pool.num_fields == 4)
            & (not (tensor_pool.fields[3] is d))
            & (tensor_pool.fields[3] == d).all()
        )
        print(f'add_field(): {flag_add_field}.')

        tensor_pool.slice(indices=[0, 1, 3])
        flag_slice = (tensor_pool.fields[3] == d[[0, 1, 3]]).all()
        print(f'slice(): {flag_slice}.')

        tensor_pool.overwrite(
            indices=[1],
            new_data=[
                torch.zeros(size=[1, 3, 2]),
                torch.zeros(size=[1, 1]),
                torch.zeros(size=[1]),
                torch.zeros(size=[1, 2]),
            ]
        )
        flag_overwrite = torch.tensor([(t == 0).all() for t in tensor_pool[1]]).all()
        print(f'overwrite(): {flag_overwrite}.')

    def test_NdArrayPool():
        print(' Test for NdArrayPool: '.center(80,'#'))

        a = [['A', 0], ['B', 1], ['C', 2], ['D', 3], ['E', 4]]
        b = 1000*np.arange(len(a))

        def getitem_a(x):
            x_1, x_2 = x
            return x_1, int(x_2)

        pool = NdArrayPool(
            fields=[a, b],
            getitems=[getitem_a, None],
        )

        flag_init = (
            isinstance(pool.fields[0], np.ndarray) 
            & isinstance(pool.fields[1], np.ndarray)
            & (not (pool.fields[0] is a))
            & (not (pool.fields[1] is b))
            & np.all(pool.fields[0] == a)
            & np.all(pool.fields[1] == b)
        )
        print(f'__init__: {flag_init}')

        a_0, b_0 = pool[0]
        flag_getitem = (
            np.all(a_0[0] == a[0][0])
            & (pool.fields[0][0][1] != a_0[1]) # '0' (str) != 0 (int)
            & np.all(a_0[1] == a[0][1])
            & np.all(b_0 == b[0]))
        print(f'__getitem__(): {flag_getitem}')

        a_extra = [['F', 5]]
        b_extra = [5000]
        pool.add([a_extra, b_extra])
        a_last, b_last = pool[-1]
        flag_add = (len(pool) == 6) & (
            (not (a_last is a_extra))
            & (not (b_last is b_extra))
            & (a_last[0] == a_extra[0][0])
            & (a_last[1] == a_extra[0][1])
            & (b_last == b_extra[0])
        )
        print(f'add(): {flag_add}')

        pool.delete(indices=[0, 5])
        flag_delete = (len(pool) == 4) & (
            np.all(pool.fields[0] == a[1:])
            & np.all(pool.fields[1] == b[1:])
        )
        print(f'delete(): {flag_delete}.')

        b_temp = pool.fields[1].copy()
        pool.shuffle(indices=[0, 1, 2])
        flag_shuffle = (
            np.any(pool.fields[1][0:3] != b_temp[0:3])
            & np.all(pool.fields[1][3] == b_temp[3])
        )
        print(f'shuffle(): {flag_shuffle} - May need multiple runs.')

        c = 0.001*(np.arange(4) + 1)
        pool.add_field(
            field=c,
            transform=str,
        )
        flag_add_field = (
            (pool.num_fields == 3)
            & (not (pool.fields[2] is c))
            & np.all(pool.fields[2] == c)
            & ([pool[i][2] for i in range(len(pool))] == [str(v) for v in c])
        )
        print(f'add_field(): {flag_add_field}.')

        pool.slice(indices=[0, 1, 3])
        flag_slice = ([pool[i][2] for i in range(len(pool))] == [str(v) for v in c[[0, 1, 3]]])
        print(f'slice(): {flag_slice}.')

        pool.overwrite(indices=[1], new_data=[[['B', 1]], [1000], [0.001]])
        flag_overwrite = torch.tensor([
            pool[1][0] == ('B', 1),
            (pool[1][1] == 1000).item(),
            pool[1][2] == '0.001',
        ]).all()
        print(f'overwrite(): {flag_overwrite}.')

    test_TensorPool()
    test_NdArrayPool()