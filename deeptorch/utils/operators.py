from abc import ABC, abstractmethod
import numpy as np

class Operator(ABC):
    def __init__(self, name):
        self.name = name

    @classmethod
    @abstractmethod
    def forward():
        pass

    @classmethod
    @abstractmethod
    def backward():
        pass
    
    @classmethod
    def __call__(self,*inputs):
        return self.forward(*inputs)

    def __str__(self):
        return f"{self.name} Operator"

    def __repr__(self):
        return f"{self.name} Operator"

class Add(Operator):
    def __init__(self, name: str = 'Add'):
        super().__init__(name)

    def forward(x, y):
        return x + y

    def backward(x, y, ref_gradient = None):
        return (np.ones_like(x), np.ones_like(y))

class Sub(Operator):
    def __init__(self, name: str = 'Sub'):
        super().__init__(name)

    def forward(x, y):
        return x - y

    def backward(x, y, ref_gradient = None):
        return (np.ones_like(x), -np.ones_like(y))

class Mul(Operator):
    def __init__(self, name: str='Mul'):
        super(). __init__(name)

    def forward(x,y):
        return x*y

    def backward(x,y,ref_gradient = None):
        return (y,x)

class MatMul(Operator):
    def __init__(self, name: str='MatMul'):
        super().__init__(name)
    
    def forward(x,y):
        return x@y
    
    def backward(x, y,ref_gradient = None):
        return (y.transpose(), x.transpose())
    

### TESTS ###

import unittest

class TestAdd(unittest.TestCase):
    def test_forward(self):
        self.assertEqual(Add()(10,12), 22, "Numerical forward failed.")
        self.assertTrue(np.array_equal(Add()(np.array([[2,4],[10,34]]),np.array([[10,3],[-3,-13]])),np.array([[12,7],[7,21]])))
    
    def test_backward(self):
        self.assertEqual(Add.backward(10,12), (1,1), "Numerical backward failed.")
        self.assertTrue(np.array_equal(Add.backward(np.array([[2,4],[10,34]]),np.array([[10,3],[-3,-13]])),(np.ones((2,2)),np.ones((2,2)))))

if __name__=="__main__":
    unittest.main(verbosity=2)