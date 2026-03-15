from graphEntity import graphEdge
# TODO: Setup functionality later

class graphEdge:
    def __init__(self,head=None,tail=None,ctxt_forward=None,ctxt_backward=None):
        self.head=head
        self.tail=tail
        self.ctxt_forward = ctxt_forward
        self.ctxt_backward = ctxt_backward
    
    def __set_forward_context__(self,ctxt_forward=None):
        self.ctxt_forward = ctxt_forward
    
    def __set_backward_context__(self,ctxt_backward=None):
        self.ctxt_backward = ctxt_backward
    
    def __set_head__(self,head=None):
        self.head=head
    
    def __set_tail__(self,tail=None):
        self.tail=tail
    