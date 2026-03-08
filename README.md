**A Tiny Deep Learning Stack Built From First Principles**

# Version 1:

### A simple architecture with 2 major components: 
1. Node: Defines the class structure for both the Data and Operations (Functions)
2. Autograph: Defines autodiff on the ValueNode and FunctionNode definitions.

### Major Problems:
1. Tightly couples the ML training execution with the Autograd because the definitions are built in such a way that the data lives in Nodes where the operations also live. Issue becomes serious once we progress to the ONNX conversion stage.
2. Current architecture does not allow for multiple children and parents. 
3. Because the definitions are managed only for one parent and one child, the backprop does not allow for a graph, instead a single chain.
4. Graph definition as a class variable is too vague, and will not allow for multiple graph creations, and does not use topological sorting, but just stores the chained order.
5. The computations and values both live on the Node and the backprop or edges are implicit. Looks fine now, but will be problematic to propagate for this defintion for an upscale or different definition.

### Summary of Version 1: 
Constrained version of operations for backpropagation of values, nodes and computations. Also, is not a worthy structure for ONNX conversion later on.


# Version 2:

### A more complex structure with multiple major components: Graph-driven IR representation oriented components
1. Tensor:
2. Parameter:
3. Operators: 
4. GraphEntity:
5. GraphEdge:
6. DAG:

### Higher-level API:
1. 

### TODO:
1. deeptorch/utils/dtypes.py
2. Build custom context manager: (See here for reference)[https://book.pythontips.com/en/latest/context_managers.html]
3. Graph IR sources: (Source 1 - paper)[https://grothoff.org/christian/teaching/2007/3353/papers/click95simple.pdf], (ONNX-Repo)[https://onnx.ai/onnx/repo-docs/IR.html#extensible-computation-graph-model]
4. Dunder methods for reference - (All dunder methods)[https://www.pythonmorsels.com/every-dunder-method/] 