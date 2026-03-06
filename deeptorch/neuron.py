from typing import Union
import torch
import pandas as pd
import numpy as np
from deeptorch.utils import *
from deeptorch.exceptions import *

class Neuron:
    def __init__(self, in_features:int, out_features:int = 1):
        self.in_f = in_features 
        self.out_f = out_features
        self.w = np.random.randn(self.in_f,self.out_f) # m x 1 weight matrix for m features and 1 output value
        self.b = np.random.random(self.out_f) # 1 x 1 bias matrix for the neuron

    def fit(self,X: Union[np.array, torch.tensor, pd.DataFrame], y: Union[np.array, torch.tensor, pd.DataFrame], 
            n_epochs:int = 100, lr:float = 1e-4, verbose: bool = False, 
            frequency: int = 5):
        """ 
        Function to fit the given dataset with the Neuron's Sigmoid function and train according to the given hyperparameters.
        Uses Standard Gradient Descent.
        """
        Y = y.reshape(-1,1) # to make the dimensions n x 1 as a 2D matrix instead of a 1D array
        for epoch in range(n_epochs):
            y_pred = self.predict_proba(X)
            loss = cross_entropy_loss(Y,y_pred)
            loss_der = cross_entropy_loss_derivative(Y,y_pred)
            w = self.w - lr*(X.T)@(loss_der)
            b = self.b - lr*np.sum(loss_der)/X.shape[0]
            self.w = w
            self.b = b
            if epoch%frequency == 0 or verbose:
                print(f"Epoch: {epoch}, loss: {loss}")

    def predict(self, X: Union[np.array, torch.tensor, pd.DataFrame]):
        if len(X.shape) > 2:
            raise DimensionMisMatchError(f"X is 3 dimensional with {X.shape}, expected 2 dimensional.")
        if not self.w.shape[0] in X.shape:
            raise MatMulError(f"Shape mismatch: Matrices with sizes {self.w.shape} cannot be multiplied with {X.shape}") 
        try:
            if self.w.shape[0] == X.shape[1]:
                return np.array(sigmoid(X@self.w+self.b) >=0.5, dtype=float) # returns n x 1 array
            elif self.w.shape[0] == X.shape[0]:
                return np.array(sigmoid(X.T@self.w+self.b) >=0.5, dtype=float) # return n x 1 array
        except Exception as e:
            print(f"Receiving error : {e}")
            return None
    
    def predict_proba(self, X: Union[np.array, torch.tensor, pd.DataFrame]):
        if len(X.shape) > 2:
            raise DimensionMisMatchError(f"X is 3 dimensional with {X.shape}, expected 2 dimensional.")
        if not self.w.shape[0] in X.shape:
            raise MatMulError(f"Shape mismatch: Matrices with sizes {self.w.shape} cannot be multiplied with {X.shape}") 
        try:
            if self.w.shape[0] == X.shape[1]:
                return sigmoid(X@self.w+self.b) # returns n x 1 array
            elif self.w.shape[0] == X.shape[0]:
                return sigmoid(X.T@self.w+self.b) # return n x 1 array
        except Exception as e:
            print(f"Receiving error : {e}")
            return None