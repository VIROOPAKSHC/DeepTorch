from abc import ABC
import numpy as np
from tensor import Tensor
"""
backward() always returns a list of gradients with respect to each of the operands.
"""

def backprop_shape_conversion(current,target):
    if (current.shape[1] == target.shape[0]) and len(target.shape)==1:
        return current.sum(axis=0)
    else:
        print("Not possible to reshape")
        return current

class Operator(ABC):
    def __init__(self, name):
        self.name = name
        self.req_operands = None
    
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

    def backward(self,x, y, forward_compute,curr_agg_grad):
        # TODO: Handle Bias's reduced over the batch axis
        inputs = [x,y]
        grads = [curr_agg_grad, curr_agg_grad]
        grads = [backprop_shape_conversion(grad,target) for (grad,target) in zip(grads,inputs)]
        return grads

class Sub(Operator):
    def __init__(self, name: str = 'Sub'):
        super().__init__(name)
        self.req_operands = 2

    def forward(x, y):
        return x - y

    def backward(self,x, y, forward_compute ,curr_agg_grad):
        # TODO: Handle Bias's reduced over the batch axis
        grads= [curr_agg_grad, -curr_agg_grad]
        inputs = [x,y]
        grads = [backprop_shape_conversion(grad,target) for (grad,target) in zip(grads,inputs)]
        return grads

class Mul(Operator):
    def __init__(self, name: str='Mul'):
        super(). __init__(name)
        self.req_operands = 2
    def forward(x,y):
        return x*y

    def backward(self,x,y,forward_compute,curr_agg_grad):
        return [curr_agg_grad*y,curr_agg_grad*x]

class MatMul(Operator):
    def __init__(self, name: str='MatMul'):
        super().__init__(name)
        self.req_operands = 2

    def forward(x,y):
        return x@y
    
    def backward(self,x, y,forward_compute,curr_agg_grad):
        return [curr_agg_grad@y.transpose(), x.transpose()@curr_agg_grad]

class Sigmoid(Operator):
    def __init__(self,name:str='Sigmoid'):
        super().__init__(name)
        self.req_operands = 1

    def forward(x):
        return 1/(1+np.exp(-x))
    
    def backward(self, x, forward_compute,curr_agg_grad):
        # print(forward_compute)
        return [(forward_compute*(1-forward_compute))*curr_agg_grad]

class Broadcast(Operator):
    def __init__(self,name:str='Broadcast'):
        super().__init__(name)
        self.req_operands=1
    
    def forward(x,shape):
        return np.broadcast_to(x,shape)

    def backward(self,x,shape,forward_compute,curr_agg_grad):
        return [curr_agg_grad]


class ReLU(Operator):
    def __init__(self,name:str='ReLU'):
        super().__init__(name)
        self.req_operands=1
    
    def forward(x):
        return np.max(np.stack(x,np.zeros_like(x)),axis=0) # TODO: Change this

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