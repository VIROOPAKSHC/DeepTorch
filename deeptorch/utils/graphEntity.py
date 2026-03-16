from operators import Operator
from tensor import Parameter, Tensor
import unittest
from exceptions import *

POSSIBLE_TYPES = ['operator','data','param']
class graphEntity:
    def __init__(self, object=None,name:str=None,ctxt_forward=None,type:str=None):
        """The function signature will be converted into *args,**kwargs
        in the future."""
        # TODO: Convert the initialization signature to accept different params as args and kwargs.
        if type in POSSIBLE_TYPES:
            self.type=type
        else:
            if isinstance(object,Parameter):
                self.type='param'
            elif isinstance(object,Tensor):
                self.type='data'
            elif isinstance(object,Operator):
                self.type='operator'
            else:
                raise ArgumentInvalidError('Passed invalid object type to initialize a graphEntity')        
        
        if self.type=='operator':
            self.__init_operator(object,name,ctxt_forward)
        else:
            self.__init_data(object,name,ctxt_forward)

    def init_object(self,object):
        if isinstance(self.object,Operator) or isinstance(self.object,Tensor):
            print("Cannot re-initialize the object!")
            return
        self.object = object

    def __init_operator(self,object=None,name:str=None,ctxt_forward=None):
        # TODO: Once initialization signature is converted, this will be converted too.
        if not isinstance(object,Operator):
            raise ArgumentInvalidError("Passed non-Operator object for operator node initialization.")
        self.object = object
        self.name = name
        self.ctxt_forward = ctxt_forward
        self.forward_compute=None

    def __init_data(self,object=None,name:str=None,ctxt_forward=None):
        # TODO: Once initialization signature is converted, this will be converted too.
        if not isinstance(object,Tensor):
            raise ArgumentInvalidError("Passed non-Tensor object for data node initialization.")
        self.object = object
        self.name = name
        self.ctxt_forward=ctxt_forward
        self.forward_compute=self.object

    def forward(self, input_edges):
        ## ORDER OF OPERATIONS WHILE BUILDING THE GRAPH IS IMPORTANT
        if not self.type=='operator':
            self.forward_compute = self.object
            return self.forward_compute
        if not self.object.req_operands == len(input_edges):
            raise GraphPropagationError(f"Not Equal Operands received for the computation. {self.object} expects {self.object.req_operands}, but received: {len(input_edges)}")
        input_args = []
        for tensor_ref in input_edges:
            input_args.append(tensor_ref.forward_compute)
        self.forward_compute = self.object(*input_args)
        return self.forward_compute

    def backward(self, input_edges):
        ## ORDER OF OPERATIONS WHILE BUILDING THE GRAPH IS IMPORTANT
        # TODO: Properly implement the backward() function and the arguments accordingly - think about scalability
        if self.type=='data':
            self.backward_compute = ctxt_backward
            return self.backward_compute
            # TODO: Handle receiving and passing backward computation
        
        elif self.type == 'param':
            # TODO: Handle parameter backward
            pass
        if not self.object.req_operands == len(input_edges):
            raise GraphPropagationError(f"Not Equal Operands received for the computation. {self.object} expects {self.object.req_operands}, but received: {len(input_edges)}")
        if not hasattr(self,'forward_compute'):
            self.forward()
        
        self.backward_compute = self.object.backward(*input_edges,self.forward_compute)
        return self.backward_compute
    
    def __str__(self):
        return f"{self.type} Node - {self.name}"

    def __repr__(self):
        return str(self)

class TestGE(unittest.TestCase):
    def __init__(self):
        pass

    def test_data(self):
        pass

    def test_operator(self):
        pass
