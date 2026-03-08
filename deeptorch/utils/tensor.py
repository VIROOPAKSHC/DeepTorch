from typing import Union, Self 
import numpy as np
import pandas as pd
import torch
from exceptions import *
import unittest

class Tensor:
    def __init__(self, value: Union[int, np.array, pd.Series, torch.Tensor], dtype: str = None, shape: Union[tuple, list] = None, type: str = 'constant'):
        """
        Arguments:
        type = Any['param', 'constant', 'input', 'output']
        only when the value is 'param',the object will store and update the gradient during the computation. 
        """
        if isinstance(value,int): 
            value=np.array(value)
            dtype = 'int'
        if isinstance(value,np.ndarray):self.value = value
        elif isinstance(value, pd.Series): self.from_pandas(value)
        elif isinstance(value, torch.Tensor): self.from_torch(value)
        else:
            self.value = value
        if dtype is not None:
            self.dtype = str(dtype)
        else:
            self.dtype = str(self.value.dtype)
 
        self.shape = shape if shape is not None else self.value.shape
        self.grad = None
        self.type = type
        
    def from_pandas(self,value: pd.Series):
        self.value = value.to_numpy()
        return self.value
    
    def from_torch(self, value: torch.Tensor):
        print(value)
        print(type(value),value.dtype)
        self.value = value.numpy()
        return self.value

    def shape(self):
        return self.value.shape
    
    def __str__(self):
        return f"{self.value!r}"

    def __repr__(self):
        return str(self)
    
    def __add__(self, other: Self):
        return Tensor(self.value + other.value,dtype=self.dtype, shape=self.shape,type=self.type)

    def __radd__(self, other: Self):
        return Tensor(other.value + self.value,dtype=other.dtype, shape=other.shape, type=other.type)

    def __sub__(self, other: Self):
        return Tensor(self.value - other.value,dtype=self.dtype, shape=self.shape, type=self.type)
    
    def __rsub__(self, other: Self):
        return Tensor(other.value - self.value, dtype=other.dtype, shape=other.dtype, type = other.type)

    def __mul__(self, other: Self):
        # This is a dot product, not the matrix multiplication
        if self.shape == other.shape:
            prod = self.value*other.value
            return Tensor(prod, dtype = self.dtype, shape = prod.shape, type = self.type)
        else:
            pass
            # TODO: Handle the mismatch in the shapes
    
    def __rmul__(self, other: Self):
        if other.shape == self.shape:
            prod = other.value*self.value
            return Tensor(prod, dtype = other.dtype, shape = prod.shape, type = other.type)
        else:
            pass
            # TODO: Handle the mismatch in the shapes

    def __matmul__(self, other: Self):
        # This method is invoked when @ is called.
        if self.shape[-1] == other.shape[0]:
            prod = self.value@other.value
            return Tensor(prod, dtype=self.dtype, shape=prod.shape, type=self.type)
        else:
            pass
            # TODO: Handle the shape mismatch
    
    def __rmatmul__(self, other: Self):
        if other.shape[-1] == self.shape[0]:
            prod = other.value @ self.value
            return Tensor(prod, dtype=other.dtype, shape=prod.shape, type=other.type)
        else:
            pass
            # TODO : Handle mismatch in shape
    
    def __eq__(self, other:Self):
        return (self.value==other.value) and (self.type==other.type) and (self.dtype==other.dtype) and (self.grad == other.grad) and (self.shape==other.shape)

    def transpose(self):
        self.value = self.value.T
        self.shape = self.value.shape
        return self

class Parameter(Tensor):
    def __init__(self, value: Union[Tensor, np.array, pd.Series, torch.Tensor]=None, dtype: str=None, shape: Union[tuple, list]=None, trainable: bool = True):
        if value is None:
            if shape is not  None:
                super().__init__(np.random.randn(*shape), dtype=dtype, shape=shape, type = 'param')
            else:
                raise ArgumentInvalidError('Both value and shape were not provided.') # should raise an argument invalid error
        else:
            super().__init__(value,dtype,shape)
        
        self.trainable = trainable


class TestTensor(unittest.TestCase):
    def test_add(self):
        pass
        # t1 = Tensor(np.array([10]))
        # t2 = Tensor(np.array([20]))
        # print(t1+t2)
        # t3 = Tensor(np.array([30]),dtype='int',shape=(1,),type='constant')
        # print(t3)
        # self.assertTrue(t1+t2 == t3)
        

if __name__=="__main__":
    unittest.main(verbosity=2)