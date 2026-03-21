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
        self.edges = defaultdict(list)
        self.nodes = set()
        self.backedges = defaultdict(list)
        self.in_count = defaultdict(int)
        self.out_count = defaultdict(int)
        self.backprop = defaultdict(list)
        self.backward_cache = {}
        self.__version = 0
        if isinstance(root,graphEntity):
            self.__register_entity(root)
    
    def add_edge(self,head:graphEntity, tail:graphEntity):
        # TODO: Modify to graphEdge object based adding later
        if head not in self.nodes:
            self.__register_entity(head)
        if tail not in self.nodes:
            self.__register_entity(tail)
        
        if not (tail in self.edges[head]):
            self.edges[head].append(tail)
            self.in_count[tail]+=1
            self.out_count[head]+=1
        
        if not (head in self.backedges[tail]):
            self.backedges[tail].append(head)

    def remove_edge(self,head,tail):
        # TODO: 
        pass
        
    def __register_entity(self,entity:graphEntity):
        if entity in self.edges and (entity in self.in_count) and (entity in self.out_count) and (entity in self.nodes):
            # TODO: Throw Warning saying the entity is already present in the dag.
            return
        self.edges[entity]=[]
        self.in_count[entity]=0
        self.out_count[entity]=0
        self.nodes.add(entity)

    def __build_forward(self):
        # Using Kahn's algorithm to perform topological sort.

        # TODO: Build a graph with list of list of nodes, where each list has nodes that can be performed in parallel.
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

        reverse_topo = self.topo_order[::-1]
        
        for i in range(len(reverse_topo)):    
            node = reverse_topo[i]
            curr_agg_grad = self.backward_cache.get(node,None)
            if curr_agg_grad is None:
                shape = node.forward_compute.shape
                curr_agg_grad = np.ones(shape)
            
            if node.type!='operator':
                node.backward([],curr_agg_grad)
                continue
            
            _ = node.backward([parent.forward_compute for parent in self.backedges[node]],curr_agg_grad)
            # assert backward_compute == node.backward_compute
            parents = self.backedges[node]
            for j in range(len(parents)):
                self.backward_cache[parents[j]] = self.backward_cache.get(parents[j],np.zeros_like(node.backward_compute[j])) + node.backward_compute[j]
            
        
            
    
if __name__=="__main__":
    dag = DAG()
    X = ge(Tensor(np.array([[1,2,5],[4,5,8],[10,3,-1],[-8,40,4]])),name='X')
    W = ge(Parameter(np.array([[0.5,0.1],[-0.5,0.3],[0.9,-0.5]])),name='W')
    mul1 = ge(MatMul(),name='MatMul(X,W)')
    b = ge(Parameter(np.array([3,-1])),name='B')
    # TODO: How to handle the implicit shape conversions in the backward.
    add1 = ge(Add(),name='Add(XW+B)')
    sigmoid = ge(Sigmoid(),name='Sigmoid(XW+B)')
    dag.add_edges([X,mul1],[W,mul1],[mul1,add1],[b,add1],[add1,sigmoid])

    dag.forward()
    dag.backward()
    for node in dag.topo_order:
        if node.type=='data':continue
        print(node)
        print(f"forward compute: {node.forward_compute}")
        print(f"backward compute: {node.backward_compute}")
        print()

        
    # A = ge(Tensor(np.array([[4,5],[6,8]])),name='A')
    # B = ge(Tensor(np.array([[-1,-2],[-3,-4]])),name='B')
    # C = ge(Tensor(np.array([[0,-4.3],[1.3,0.03]])),name='C')
    # mul = ge(MatMul(),name='matmul(AB)')
    # sum1 = ge(Add(),name='AB+C')
    # dag.add_edges([A,mul],[B,mul],[mul,sum1],[C,sum1])
    