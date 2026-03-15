# from graphEdge import graphEdge
from graphEntity import graphEntity as ge
from graphEntity import graphEntity
from tensor import Tensor, Parameter
import numpy as np
from operators import *
from collections import defaultdict
from exceptions import *

class DAG:
    
    def __init__(self,root:graphEntity=None):
        if isinstance(root,graphEntity):
            self.__register_root__(root)
        self.dag = defaultdict(list)
        self.backprop = defaultdict(list)
        if isinstance(root,graphEntity):
            self.add_entity(root)
    
    def register_edge(self,head:graphEntity, tail:graphEntity):
        # TODO: Modify to graphEdge object based adding later
        if head not in self.dag:
            self.register_entity(head)
        if tail not in self.dag:
            self.register_entity(tail)
        
        self.dag[head].append(tail)

        if head.type == 'operator':
            if not hasattr(self,'root'):
                self.__register_root__(head)

            head.__add_operator_output__(tail)
    
        if tail.type == 'operator':
            if not hasattr(self,'root'):
                self.__register_root__(tail)

            tail.__add_operator_input__(head)

    def __register_root__(self,entity:graphEntity):
        if not hasattr(self,'root') and entity.type == 'operator': 
            # len(self.dag)==0 was earlier taken as well. Is it really needed? TODO: Find out!
            self.dag[entity]=[]
            self.root = entity
        else:
            print("Data node cannot be registered as node.")
        
    def register_entity(self,entity:graphEntity):
        if entity in self.dag:
            # TODO: Throw Warning saying the entity is already present in the dag.
            return
        self.dag[entity]=[]
    
    def __build_forward__(self):
        # Instead of BFS, perform Topological Sort
        queue = []
        visited = {}
        queue.append(self.root)
        while len(queue):
            node = queue.pop(0)
            if visited.get(node,False)==True:
                continue
            visited[node]=True
            if node.type=='operator':
                for input in node.input_tensor_refs:
                    queue.append(input)
                node.forward()
            else:
                node.forward()  
            
            for child in self.dag[node]:
                if not visited.get(child,False):
                    queue.append(child)
         
    def register_edges(self,*args):
        """
        Provide as a list of tuples or lists, with 2 items each where first one is a head
        and the second one is a tail
        """
        for l in args:
            if len(l)!=2:
                raise ArgumentInvalidError('Expected 2 items as head and tail for each list to register as edges.')
            self.register_edge(*l)

    def forward(self):
        self.__build_forward__()

    def __build_backward__(self):
        self.backward_graph = defaultdict(list)
        for node in self.dag:
            if node.type!='operator':continue
            for child in node.input_tensor_refs:
                self.backward_graph[node].append(child)

    def __find_leaves__(self):
        if not hasattr(self,'backward_graph'):
            self.__build_backward__()
        leaves = []
        for node, children in self.dag.items():
            if len(children)==0:
                leaves.append(node)
        self.leaves=leaves

    def backward(self):
        self.__build_backward__()
        self.__find_leaves__()
        


if __name__=="__main__":
    dag = DAG()
    X = ge(Tensor(np.array([[1,2,5],[4,5,8],[10,3,-1],[-8,40,4]])),name='X')
    W = ge(Parameter(np.array([[0.5,0.1],[-0.5,0.3],[0.9,-0.5]])),name='W')
    mul1 = ge(MatMul(),name='MatMul(X,W)')
    b = ge(Parameter(np.array([3,-1])),name='B')
    add1 = ge(Add(),name='Add(XW+B)')
    sigmoid = ge(Sigmoid(),name='Sigmoid(XW+B)')
    dag.register_edges([X,mul1],[W,mul1],[mul1,add1],[b,add1],[add1,sigmoid])

    # Condensed all of the following into one line as above.
    # dag.register_edge(X,mul1)
    # dag.register_edge(W,mul1)
    # dag.register_edge(mul1,add1)
    # dag.register_edge(b,add1)
    # dag.register_edge(add1,sigmoid)

    # print('Graph as Adjacency List: \n',dag.dag)
    dag.forward()
    # for node in dag.dag:
    #     if node.type!='operator':continue
    #     print(f"{node}:, {node.input_tensor_refs}, {node.output_tensor_refs}")
    #     print(f"forward compute: {node.forward_compute}")
    dag.backward()
    print(dag.backward_graph)
    print(dag.leaves)

    