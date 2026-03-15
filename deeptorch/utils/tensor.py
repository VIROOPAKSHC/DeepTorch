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
    
    def args(self):
        return {'value':self.value,'dtype':self.dtype,'shape':self.shape,'type':self.type}
    
    def __str__(self):
        return f"{self.value!r}"

    def __repr__(self):
        return str(self)
    
    def __add__(self, other: Self):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        return Tensor(self.value + other.value,dtype=self.dtype, shape=self.shape,type=self.type)
    
    def __radd__(self, other: Self):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        return Tensor(other.value + self.value,dtype=other.dtype, shape=other.shape, type=other.type)

    def __sub__(self, other: Self):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        return Tensor(self.value - other.value,dtype=self.dtype, shape=self.shape, type=self.type)
    
    def __rsub__(self, other: Self):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        return Tensor(other.value - self.value, dtype=other.dtype, shape=other.dtype, type = other.type)

    def __mul__(self, other: Self):
        # This is a dot product, not the matrix multiplication
        if not isinstance(other,Tensor):
            other = Tensor(other)
        if self.shape == other.shape:
            prod = self.value*other.value
            return Tensor(prod, dtype = self.dtype, shape = prod.shape, type = self.type)
        else:
            pass
            # TODO: Handle the mismatch in the shapes
    
    def __rmul__(self, other: Self):
        if not isinstance(other,Tensor):
            other = Tensor(other)
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
    
    def __truediv__(self, other: Union[Self,np.array,int,float]):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        return Tensor(self.value/other.value)
    
    def __rtruediv__(self, other: Union[Self,np.array, int, float]):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        return Tensor(other.value/self.value)

    def inv(self, inplace:bool =False):
        # TODO: should allow for inplace
        kwargs = self.args()
        kwargs['value'] = 1/self.value
        a = Tensor(**kwargs)
        if inplace:self.value = a.value
        return a

    def __abs__(self):
        # TODO: should allow for inplace
        kwargs = self.args()
        kwargs['value'] = np.abs(self.value)
        return Tensor(**kwargs)
    
    def __round__(self, places=None):
        # TODO: should allow for inplace
        kwargs = self.args()
        kwargs['value'] = np.round(self.value,places)
        return Tensor(**kwargs)

    def __eq__(self, other:Self):
        return np.allclose(self.value,other.value) 

    def transpose(self):
        # TODO: should allow for inplace
        kwargs = self.args()
        kwargs['value'] = self.value.T
        kwargs['shape'] = self.value.shape
        return Tensor(**kwargs)
    
    def __neg__(self):
        kwargs = self.args()
        kwargs['value']=-self.value
        return Tensor(**kwargs)
    
    def __pos__(self):
        kwargs = self.args()
        return Tensor(**kwargs)
    
    def __array_ufunc__(self,ufunc, method, *inputs, **kwargs):
        # TODO: Not handling method, *inputs, **kwargs
        d = self.args()
        d['value'] = ufunc(self.value)
        return Tensor(**d)

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

    def update_grad(self):
        
        # TODO: Write code to update gradient based on the optimizer
        pass

class TestTensor(unittest.TestCase):
    def test_init(self):
        t = Tensor(10)
        self.assertTrue((t.value==10))
        print(t.shape)
        self.assertTrue(t.shape==(1,))
        self.assertTrue(t.dtype=='int')
        self.assertTrue(t.type=='constant')
        t1 = Tensor(np.random.randn(5,6),type='param')
        self.assertTrue((t1.shape == (5,6)) and (t1.dtype=='float16') and 
                        (t1.grad is None) and (t1.type == 'param'))
        t2 = Tensor(pd.Series([10,16,18,20]))
        t3 = Tensor(torch.Tensor([10,16,18,20]))


    def test_add(self):
        # pass
        t1 = Tensor(np.array([10]))
        t2 = Tensor(np.array([20]))
        t3 = Tensor(np.array([30]))
        self.assertTrue((t1+t2) == (t3))
    
    def test_invert(self):
        t1 = Tensor(np.array([2,3,4,5]))
        self.assertEqual(t1.inv(),Tensor(np.array([1/2,1/3,1/4,1/5])))

    def test_div(self):
        t1 = Tensor(np.array([3,6,78,2,4]))
        self.assertEqual(t1/t1,Tensor(np.array([1,1,1,1,1])))
    
    def test_exp(self):
        t1 = Tensor(np.array([0,0.5,1,100]))
        self.assertAlmostEqual(1/(1+np.exp(-t1)),Tensor(np.array([1/(1+np.exp(-0.0)),1/(1+np.exp(-0.5)),1/(1+np.exp(-1)),1/(1+np.exp(-100))])))

if __name__=="__main__":
    unittest.main(verbosity=2)