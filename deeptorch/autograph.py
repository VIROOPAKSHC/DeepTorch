from typing import Union, Self
import numpy as np
from deeptorch.node import Node
from deeptorch.utils.exceptions import *

class Value(Node):
    def __init__(self,value: Union[int, float, np.array] = None, name:str=None):
        super().__init__("value")
        self.value = value if type(value)!=Node else value.value
        if name:self.name=name
        else:
            self.name=f"var_{Node.NAMEVARCOUNTER}"
            Node.NAMEVARCOUNTER+=1
        print(f"{self.name} variable has been created")
    
    def __add__(self,right: Union[Self, int, float,np.array]):
        add = Add(self,right)
        add.apply()
        self.next = add
        if isinstance(right,Value):right.next = add
        output = add.output
        add.next = output
        output.prev = add
        return output
    
    def __mul__(self,right: Union[Self, int, float,np.array]):
        mul = Multiply(self,right)
        mul.apply()
        output = mul.output
        self.next = mul
        mul.next = output
        output.prev = mul
        if isinstance(right,Value):right.next = mul
        return output
    
    def __rmul__(self,right:Union[Self, int, float,np.array]):
        output = right*self
        return output

    def __sub__(self,right: Union[Self, int, float,np.array]):
        sub = Subtract(self,right)
        sub.apply()
        output = sub.output
        self.next = sub
        if isinstance(right,Value):right.next = sub
        sub.next = output
        output.prev = sub
        return output
    
    def __pow__(self, right: int):
        pow = Pow(self,right)
        pow.apply()
        output = pow.output
        self.next = pow
        pow.next = output
        output.prev = pow
        return output

    def backward(self):
        if self.is_root():
            self.gradient = 1
        else:
            if self.next == None:
                raise GraphPropagationError("Not a root node, but encountered an empty next node. Formation error?")
            if self.next.type == 'function':
                self.gradient = self.gradbackprop
            elif self.next.type=='value':
                raise GraphPropagationError("Encountered a value node followed by a value node. Formation error?")
            else:
                raise GraphPropagationError("Encountered unknown error. ")
        
        if not self.is_leaf():
            self.prev.backward()

class Function(Node):
    def __init__(self,operation :str = None,name: str=None):
        super().__init__('function')
        self.left_operand = None
        self.right_operand = None
        self.operation = operation
        self.value = operation
        if name:self.name=name
        else:
            self.name=f"func_{Node.NAMEFUNCTIONCOUNTER}"
            Node.NAMEFUNCTIONCOUNTER+=1
        print(f"{self.name} function {self.operation} has been created")

class Add(Function):
    def __init__(self, left_operand: Union[Value, int, float,np.array], right_operand: Union[Value, int, float], output: Union[Value, int, float, np.array] = None):
        super().__init__('+')
        self.left_operand=left_operand
        self.right_operand=right_operand
        self.output = output if isinstance(output,Value) else Value(output)
    
    def apply(self):
        if isinstance(self.right_operand,Value):
            right = self.right_operand.value
        else:
            right = self.right_operand
        self.output.value = (self.left_operand.value + right)
    
    def backward(self):
        self.left_operand.gradbackprop = 1*self.next.gradient
        self.left_operand.backward()

        if isinstance(self.right_operand,Value):
            self.right_operand.gradbackprop = 1*self.next.gradient
            self.right_operand.backward()
        

class Subtract(Function):
    def __init__(self, left_operand: Union[Value, int, float,np.array], right_operand: Union[Value, int, float], output: Union[Value, int, float, np.array] = None):
        super().__init__('-')
        self.left_operand=left_operand
        self.right_operand=right_operand
        self.output = output if isinstance(output,Value) else Value(output)
    
    def apply(self):
        if isinstance(self.right_operand,Value):
            right = self.right_operand.value
        else:
            right = self.right_operand
        self.output.value = self.left_operand.value - right
    
    def backward(self):
        self.left_operand.gradbackprop = 1*self.next.gradient
        self.left_operand.backward()
        if isinstance(self.right_operand,Value):
            self.right_operand.gradbackprop = -1*self.next.gradient
            self.right_operand.backward()

class Multiply(Function):
    def __init__(self, left_operand: Union[Value, int, float,np.array], right_operand: Union[Value,int, float], output: Union[Value, int, float, np.array] = None):
        super().__init__('*')
        self.left_operand=left_operand
        self.right_operand=right_operand
        self.output = output if isinstance(output,Value) else Value(output)
    
    def apply(self):
        if isinstance(self.right_operand,Value):
            right = self.right_operand.value
        else:
            right = self.right_operand
        self.output.value = self.left_operand.value*right
    
    def backward(self):
        if type(self.right_operand)!=Value:
            right = self.right_operand
        else:
            right = self.right_operand.value
            self.right_operand.gradbackprop = self.left_operand.value*self.next.gradient
            self.right_operand.backward()
        
        self.left_operand.gradbackprop = right*self.next.gradient
        self.left_operand.backward()
        

class Pow(Function):
    def __init__(self, left_operand: Union[Value, int, float,np.array], right_operand: int, output: Union[Value, int, float, np.array] = None):
        super().__init__('**')
        self.left_operand=left_operand
        self.right_operand=right_operand
        self.output = output if isinstance(output,Value) else Value(output)
    
    def apply(self):
        self.output.value = self.left_operand.value ** self.right_operand
    
    def backward(self):
        right = self.right_operand
        self.left_operand.gradbackprop = right*(self.left_operand.value ** (right-1))*self.next.gradient
        self.left_operand.backward()

class Sigmoid(Function):
    def __init__(self,left_operand: Union[Value,int,float,np.array]=None,output: Union[Value, int, float, np.array]=None):
        super().__init__("sigmoid")
        self.left_operand = left_operand
        self.output = output if isinstance(output,Value) else Value(output)
        if left_operand is not None:self.apply()
    
    def apply(self,left_operand: Union[Value,int,float,np.array]=None):
        # Cases:
        # 1. Already passed a value while init and didnt send now
        # 2. Already passed a value while init and sent now
        # 3. Didnt pass a value while init and sent now
        if left_operand is not None:
            if not isinstance(self.left_operand,Value):
                self.left_operand = left_operand if isinstance(left_operand,Value) else None
            
            value = left_operand.value if isinstance(left_operand,Value) else left_operand
            if isinstance(left_operand,Value):
                self.prev = left_operand
                left_operand.next = self
        else:
            if self.left_operand is not None:
                if isinstance(self.left_operand,Value):
                    self.prev = left_operand
                    self.left_operand.next = self
                    value = self.left_operand.value
                else:
                    value = self.left_operand                
            else:
                raise ArgumentInvalidError("must provide a valid argument to perform this operation")

        self.output.value = 1/(1+np.exp(value))
        self.output.prev = self
        self.next = self.output
        return self.output
    
    def backward(self):
        sigm = self.output.value
        if isinstance(self.left_operand,Value):
            self.left_operand.gradbackprop = sigm*(1-sigm)
            self.left_operand.backward()