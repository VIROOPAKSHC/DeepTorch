import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_dash(x):
    return sigmoid(x)*(1-sigmoid(x))

def cross_entropy_loss(y_true, y_pred, reduction: str = 'mean'):
    """ 
    Produces the mean or sum of the Cross Entropy Loss over the batch
    """
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15) # to keep the array from having 0 or 1 values so that the log might explode or cause NaN values
    loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    if reduction == 'mean':
        return round(loss/y_true.shape[0],4)
    else:
        return loss

def cross_entropy_loss_derivative(y_true, y_pred):
    return (y_pred-y_true)

def accuracy(y_true, y_pred):
    return float((y_true.reshape(-1,1)==y_pred.reshape(-1,1)).sum()/y_pred.shape[0])*100

