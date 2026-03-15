class CustomError(Exception):
    def __init__(self,message):
        self.message = message
        super().__init__(message)

class MatMulError(CustomError):
    def __str__(self):
        return f"{self.message} has been raised when performing Matrix Multiplication"
    
class DimensionMisMatchError(CustomError):
    def __str__(self):
        return f"{self.message} has been raised when checking for dimension match"
    
class GraphPropagationError(CustomError):
    def __str__(self):
        return f"{self.message} has been raised while trying to propagate values through the graph." 

class ArgumentInvalidError(CustomError):
    def __str__(self):
        return f"{self.message} has been raised. Check Arguments."
    