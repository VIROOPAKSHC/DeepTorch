from abc import ABC, abstractmethod

class Node:
    ## No parallel processing yet. Because we do not handle Atomic operations of any variables like counters and dynamic graph
    NAMEVARCOUNTER = 1
    NAMEFUNCTIONCOUNTER = 1
    Graph = []
    def  __init__(self, type:str):
        self.name = None
        self.type = type
        self.value = None
        self.gradient = None
        self.next = None
        self.prev = None
        self.gradbackprop = None
        Node.Graph.append(self)
    
    def __str__(self):
        return f"{self.value!r}"

    def __repr__(self):
        return f"{self.value!r}"
    
    def __value__(self):
        return self.value

    def is_leaf(self):
        return self.prev==None

    def is_root(self):
        return self.next==None

    def __rsub__(self,other):
        return other-self

    def __radd__(self,other):
        return other+self