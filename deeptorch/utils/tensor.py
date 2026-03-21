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
            shape = (1,)
        if isinstance(value,np.ndarray):self.value = value
        elif isinstance(value, pd.Series): self.from_pandas(value)
        elif isinstance(value, torch.Tensor): self.from_torch(value)
        elif isinstance(value,Tensor):
            self = value
        else:
            # TODO: Handle unknown type initialization
            pass

        if dtype is not None:
            self.dtype = str(dtype)
        else:
            # print("Printing :",self.value)
            self.dtype = str(self.value.dtype)
 
        self.shape = shape if shape is not None else self.value.shape
        self.grad = None
        self.type = type
        # print("CAME TILL HERE")

    def from_pandas(self,value: pd.Series):
        self.value = value.to_numpy()
        return self.value
    
    def from_torch(self, value: torch.Tensor):
        # print(value)
        # print(type(value),value.dtype)
        self.value = value.numpy()
        return self.value
    
    def sum(self,axis=None, dtype=None, out=None, **kwargs):
        return self.value.sum(axis,dtype,out,**kwargs)

    def shape_(self):
        return self.value.shape
    
    def args(self):
        return {'value':self.value,'dtype':self.dtype,'shape':self.shape,'type':self.type}
    
    def __str__(self):
        return f"{self.value!r}"

    def __repr__(self):
        return str(self)
    
    def __add__(self, other: Self):
        # print("Types of both :",type(self),type(other))
        # print("Types of both of their values :",type(self.value),type(other.value))
        if not isinstance(other,Tensor):
            other = Tensor(other)
        return Tensor(self.value + other.value,dtype=self.dtype, shape=self.shape,type=self.type)
    
    def __radd__(self, other: Self):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        return Tensor(other.value + self.value,dtype=self.dtype, shape=self.shape, type=self.type)

    def __sub__(self, other: Self):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        return Tensor(self.value - other.value,dtype=self.dtype, shape=self.shape, type=self.type)
    
    def __rsub__(self, other: Self):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        return Tensor(other.value - self.value, dtype=self.dtype, shape=self.shape, type = self.type)

    def __mul__(self, other: Self):
        # This is a dot product, not the matrix multiplication
        if isinstance(other,Tensor):
            other = other.value
        try:
            prod = self.value*other
            return Tensor(prod, dtype = self.dtype, shape = prod.shape, type = self.type)
        except Exception as e:
            raise GraphPropagationError(f"Shape mismatch while product,{self.shape} and {other.shape}. Caused error : {e} ")
            pass
            # TODO: Handle the mismatch in the shapes
    
    def __rmul__(self, other: Self):
        if isinstance(other,Tensor):
            other = other.value
        try:
            prod = other*self.value
            return Tensor(prod, dtype = self.dtype, shape = prod.shape, type = self.type)
        except Exception as e:
            raise GraphPropagationError(f"Shape mismatch while product,{self.shape} and {other.shape}. Caused error : {e} ")
            # TODO: Handle the mismatch in the shapes

    def __matmul__(self, other: Self):
        # This method is invoked when @ is called.
        if isinstance(other,Tensor):
            other = other.value
        try:
            prod = self.value@other
            return Tensor(prod, dtype=self.dtype, shape=prod.shape, type=self.type)
        except Exception as e:
            raise GraphPropagationError(f"Invalid shapes for matmul, received : {self.value.shape} and {other.shape}. Exception received : {e}")
        

    def __rmatmul__(self, other: Self):
        if isinstance(other,Tensor):
            other=other.value
        try:
            prod = other @ self.value
            return Tensor(prod, dtype=self.dtype, shape=prod.shape, type=self.type)
        except Exception as e:
            raise GraphPropagationError(f"Invalid shapes for rmatmul, received : {self.value.shape} and {other.shape}. Exception received : {e}")
        
        
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
        if isinstance(other,Tensor):
            other = other.value
        return np.allclose(self.value,other) 

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
    
    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    def __pos__(self):
        kwargs = self.args()
        return Tensor(**kwargs)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        unwrapped = []
        for x in inputs:
            if isinstance(x, Tensor):
                unwrapped.append(x.value)
            else:
                unwrapped.append(x)

        if "out" in kwargs and kwargs["out"] is not None:
            out = kwargs["out"]
            kwargs["out"] = tuple(
                o.value if isinstance(o, Tensor) else o
                for o in out
            )

        result = ufunc(*unwrapped, **kwargs)

        if isinstance(result, tuple):
            return tuple(
                Tensor(x) if isinstance(x, np.ndarray) or np.isscalar(x) else x
                for x in result
            )

        return Tensor(result)

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
        self.assertTrue(t.shape==(1,))
        self.assertTrue(t.dtype=='int')
        self.assertTrue(t.type=='constant')
        
        t1 = Tensor(np.random.randn(5,6),type='param')
        self.assertTrue(t1.shape == (5,6))
        self.assertTrue((t1.dtype=='float64'))
        self.assertTrue(t1.grad is None)
        self.assertTrue(t1.type == 'param')

        t2 = t1
        self.assertTrue(t2.shape == (5,6))
        self.assertTrue((t2.dtype=='float64'))
        self.assertTrue(t2.grad is None)
        self.assertTrue(t2.type == 'param')
        self.assertTrue(np.equal(t2.value,t1.value).all())

    def test_shape(self):
        value = np.array([[10,30],[10,39],[-1,3]])
        t1 = Tensor(value)
        self.assertTrue(t1.shape_()==value.shape)
        
    def test_args(self):
        value = np.array([[-2,0.4,10],[4.6,9.6,2.98],[-9.31,6.85,4.23]])
        t1 = Tensor(value)
        self.assertTrue(t1.args()=={'value':value,'dtype':t1.dtype,'shape':t1.shape,'type':t1.type})

    def test_add(self):
        # pass
        v1 = np.random.randn(5,2)
        v2 = np.random.randn(5,2)
        t1 = Tensor(v1)
        t2 = Tensor(v2)
        t3 = Tensor(v1+v2)
        self.assertTrue((t1+t2) == (t3))
    
    def test_equal(self):
        value = np.array([[10,30],[10,39],[-1,3]])
        t1 = Tensor(value)
        self.assertTrue(t1==value)
        self.assertTrue(t1==t1)

    def test_invert(self):
        t1 = Tensor(np.array([2,3,4,5]))
        self.assertEqual(t1.inv(),Tensor(np.array([1/2,1/3,1/4,1/5])))

    # def test_div(self):
    #     t1 = Tensor(np.array([3,6,78,2,4]))
    #     self.assertEqual(t1/t1,Tensor(np.array([1,1,1,1,1])))
    
    def test_exp(self):
        v1 = np.array([0,0.5,1,100])
        v2 = 1/(1+np.exp(-v1))
        t1 = Tensor(v1)
        self.assertTrue(1/(1+np.exp(-t1))==v2)

if __name__=="__main__":
    unittest.main(verbosity=2)