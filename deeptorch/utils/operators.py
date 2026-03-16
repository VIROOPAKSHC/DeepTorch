from abc import ABC, abstractmethod
import numpy as np

class Operator(ABC):
    def __init__(self, name):
        self.name = name
        self.req_operands = None
    @classmethod
    @abstractmethod
    def forward():
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
        self.req_operands = 2

    def forward(x, y):
        return x + y

    def backward(self,x, y, forward_compute = None):
        return np.array([np.ones_like(x), np.ones_like(y)])

class Sub(Operator):
    def __init__(self, name: str = 'Sub'):
        super().__init__(name)
        self.req_operands = 2

    def forward(x, y):
        return x - y

    def backward(self,x, y, forward_compute = None):
        return np.array([np.ones_like(x), -np.ones_like(y)])

class Mul(Operator):
    def __init__(self, name: str='Mul'):
        super(). __init__(name)
        self.req_operands = 2
    def forward(x,y):
        return x*y

    def backward(self,x,y,forward_compute = None):
        return np.array([y,x])

class MatMul(Operator):
    def __init__(self, name: str='MatMul'):
        super().__init__(name)
        self.req_operands = 2

    def forward(x,y):
        return x@y
    
    def backward(self,x, y,forward_compute = None):
        return np.array([y.transpose(), x.transpose()])

class Sigmoid(Operator):
    def __init__(self,name:str='Sigmoid'):
        super().__init__(name)
        self.req_operands = 1

    def forward(x):
        return 1/(1+np.exp(-x))
    
    def backward(self, x, forward_compute):
        return forward_compute*(1-forward_compute)

class ReLU(Operator):
    def __init__(self,name:str='ReLU'):
        super().__init__(name)
        self.req_operands=1
    
    def forward(x):
        return np.max(np.stack(x,np.zeros_like(x)),axis=0)

    def backward(self,x, forward_compute=None):
        # TODO: How to write backward for ReLU?
        pass

class Softmax(Operator):
    def __init__(self,name:str='Softmax'):
        super().__init__(name)
        self.req_operands=1
    def forward(x):
        pass

    def backward(self,x, forward_compute=None):
        pass

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