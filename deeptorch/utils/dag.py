# from graphEdge import graphEdge
from graphEntity import graphEntity as ge
from graphEntity import graphEntity
from tensor import Tensor, Parameter
import numpy as np
from operators import *
from collections import defaultdict, deque
from exceptions import *

class DAG:
    def __init__(self,root:graphEntity=None):
        if isinstance(root,graphEntity):
            self.__register_root(root)
        self.edges = defaultdict(list)
        self.nodes = set()
        self.backedges = defaultdict(list)
        self.in_count = defaultdict(int)
        self.out_count = defaultdict(int)
        self.backprop = defaultdict(list)
        self.__version = 0
        if isinstance(root,graphEntity):
            self.__register_entity(root)
    
    def add_edge(self,head:graphEntity, tail:graphEntity):
        # TODO: Modify to graphEdge object based adding later
        if head not in self.nodes:
            self.__register_entity(head)
        if tail not in self.nodes:
            self.__register_entity(tail)
        
        self.edges[head].append(tail)
        self.in_count[tail]+=1
        self.out_count[head]+=1
        self.backedges[tail].append(head)
        if head.type=='operator':
            if not hasattr(self,'root'):
                self.__register_root(head)
        if tail.type=='operator':
            if not hasattr(self,'root'):
                self.__register_root(tail)

    def remove_edge(self,head,tail):
        # TODO: 
        pass

    def __register_root(self,entity:graphEntity):
        if not hasattr(self,'root') and entity.type == 'operator': 
            # len(self.dag)==0 was earlier checked as well. Is it really needed? TODO: Find out!
            self.edges[entity]=[]
            self.root = entity
        else:
            print("Data node cannot be registered as node.")
        
    def __register_entity(self,entity:graphEntity):
        if entity in self.edges:
            # TODO: Throw Warning saying the entity is already present in the dag.
            return
        self.edges[entity]=[]
        self.nodes.add(entity)

    def __build_forward(self):
        # Instead of BFS, perform Topological Sort
        self.topo_order = []
        in_degrees = self.in_count.copy()
        queue = deque()
        for node in self.nodes:
            if in_degrees[node]==0:
                queue.append(node)
        visited = {}
        while len(queue):
            node = queue.popleft()
            if visited.get(node,False)==True:
                raise GraphPropagationError(f"Somehow this node : {node} has repeated")
            visited[node]=True
            self.topo_order.append(node)
            for child in self.edges[node]:
                if in_degrees[child]>0:
                    in_degrees[child]-=1
                if in_degrees[child]==0:
                    queue.append(child)
        # print(self.topo_order)
         
    def add_edges(self,*args):
        """
        Provide as a list of tuples or lists, with 2 items each where first one is a head
        and the second one is a tail
        """
        for l in args:
            if len(l)!=2:
                raise ArgumentInvalidError('Expected 2 items as head and tail for each list to register as edges.')
            self.add_edge(*l)

    def forward(self):
        self.__build_forward()
        for node in self.topo_order:
            node.forward(self.backedges[node])

    def backward(self):
        if not hasattr(self,'topo_order'):
            self.forward()
        ctxt_backward = 1
        for node in self.topo_order[::-1]:
            print("Node :",node)
            ctxt_backward=node.backward(self.backedges[node],ctxt_backward)
            # =node.backward_compute= 
    
if __name__=="__main__":
    dag = DAG()
    X = ge(Tensor(np.array([[1,2,5],[4,5,8],[10,3,-1],[-8,40,4]])),name='X')
    W = ge(Parameter(np.array([[0.5,0.1],[-0.5,0.3],[0.9,-0.5]])),name='W')
    mul1 = ge(MatMul(),name='MatMul(X,W)')
    b = ge(Parameter(np.array([3,-1])),name='B')
    add1 = ge(Add(),name='Add(XW+B)')
    sigmoid = ge(Sigmoid(),name='Sigmoid(XW+B)')
    dag.add_edges([X,mul1],[W,mul1],[mul1,add1],[b,add1],[add1,sigmoid])

    # Condensed all of the following into one line as above.
    # dag.add_edge(X,mul1)
    # dag.add_edge(W,mul1)
    # dag.add_edge(mul1,add1)
    # dag.add_edge(b,add1)
    # dag.add_edge(add1,sigmoid)

    # print('Graph as Adjacency List: \n',dag.dag)
    dag.forward()
    # for node in dag.nodes:
    #     if node.type!='operator':continue
    #     print(f"{node}:, {dag.in_count[node]}, {dag.out_count[node]}")
    #     print(f"forward compute: {node.forward_compute}")
    dag.backward()
    print(dag.topo_order)

    