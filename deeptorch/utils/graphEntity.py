from operators import Operator
from tensor import Parameter, Tensor
from typing import Literal,  Self, Optional
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
            self.__init_operator__(object,name,ctxt_forward)
        else:
            self.__init_data__(object,name,ctxt_forward)

    def init_object(self,object):
        if isinstance(self.object,Operator) or isinstance(self.object,Tensor):
            print("Cannot re-initialize the object!")
            return
        self.object = object

    def __init_operator__(self,object=None,name:str=None,ctxt_forward=None):
        # TODO: Once initialization signature is converted, this will be converted too.
        if not isinstance(object,Operator):
            raise ArgumentInvalidError("Passed non-Operator object for operator node initialization.")
        self.object = object
        self.name = name
        self.ctxt_forward = ctxt_forward
        self.input_tensor_refs = []
        self.output_tensor_refs = []
        self.forward_compute=None
    
    def __add_operator_input__(self,*input_tensors):
        if not self.type=='operator':return
        if not all(map(lambda x:type(x)==graphEntity,input_tensors)):
            raise ArgumentInvalidError("Can only pass Graph Entity objects as inputs.")
        self.input_tensor_refs.extend(list(input_tensors))

    def __add_operator_output__(self,*output_tensors):
        if not self.type=='operator':return
        self.output_tensor_refs.extend(list(output_tensors))

    def __init_data__(self,object=None,name:str=None,ctxt_forward=None):
        # TODO: Once initialization signature is converted, this will be converted too.
        if not isinstance(object,Tensor):
            raise ArgumentInvalidError("Passed non-Tensor object for data node initialization.")
        self.object = object
        self.name = name
        self.ctxt_forward=ctxt_forward
        self.forward_compute=self.object

    def __add_forward_entity__(self,entity: Self):
        self.outgoing.append(entity)

    def __add_backward_entity__(self,entity: Self):
        self.incoming.append(entity)

    def forward(self):
        ## ORDER OF OPERATIONS WHILE BUILDING THE GRAPH IS IMPORTANT
        if not self.type=='operator':
            self.forward_compute = self.object
            return
        if not self.object.req_operands == len(self.input_tensor_refs):
            raise GraphPropagationError(f"Not Equal Operands received for the computation. {self.object} expects {self.object.req_operands}, but received: {len(self.input_tensor_refs)}")
        input_args = []
        for tensor_ref in self.input_tensor_refs:
            input_args.append(tensor_ref.forward_compute)
        self.forward_compute = self.object(*input_args)

    def backward(self, ctxt_backward):
        ## ORDER OF OPERATIONS WHILE BUILDING THE GRAPH IS IMPORTANT
        if self.type=='data':
            self.backward_compute = ctxt_backward
            # TODO: Handle receiving and passing backward computation
        if not self.object.req_operands == len(self.input_tensor_refs):
            raise GraphPropagationError(f"Not Equal Operands received for the computation. {self.object} expects {self.object.req_operands}, but received: {len(self.input_tensor_refs)}")
        if not hasattr(self,'forward_compute'):
            self.forward()
        self.backward_compute = self.object.backward(*self.input_tensor_refs)

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
