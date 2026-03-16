import numpy as np

try:
    from .autodiff import Node
except ImportError:
    from autodiff import Node

def mse(y_true, y_pred):
    """
    Mean Squared Error (MSE)
    y_true: array atau list dari label asli
    y_pred: objek Node dari output FFNN
    """
    y_true = np.array(y_true)
    diff = y_pred.data - y_true # error / selisih hasil prediksi vs asli
    
    N = y_pred.data.size
    loss_val = np.mean(diff ** 2)
    
    # Bungkus hasil ke dalam Node untuk AutoDiff
    out = Node(loss_val, (y_pred,))
    
    # Turunan dari MSE
    def _backward():
        y_pred.grad += (2.0 / N) * diff * out.grad # Turunan matematis d(MSE) / d(y_pred) = 2/N * (y_pred - y_true)
        
    out._backward = _backward
    return out

def categoricalCrossentropy(y_true, y_pred, from_softmax=False):
    """
    Categorical Cross-Entropy
    """
    y_true = np.array(y_true)

    eps = 1e-15
    y_pred_clipped = np.clip(y_pred.data, eps, 1 - eps)
    
    batch_size = y_pred.data.shape[0] if len(y_pred.data.shape) > 1 else 1
    
    # Rumus: -1/N * sum(y_true * ln(y_pred))
    loss_val = (-1 * np.sum(y_true * np.log(y_pred_clipped))) / batch_size
    
    out = Node(loss_val, (y_pred,))
    
    def _backward():
        if from_softmax:           
            grad = (y_pred.data - y_true) / batch_size
            y_pred.grad += grad * out.grad
        else:
            grad = (-y_true / y_pred_clipped) / batch_size
            y_pred.grad += grad * out.grad
        
    out._backward = _backward
    return out

def binaryCrossentropy(y_true, y_pred):
    """
    Binary Cross-Entropy
    """
    y_true = np.array(y_true)
    
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred.data, eps, 1 - eps)
    
    batch_size = y_pred.data.shape[0] if len(y_pred.data.shape) > 1 else 1
    
    # Rumus: -1/N * sum(y_true * ln(y_pred) + (1 - y_true) * ln(1 - y_pred))
    loss_val = (-1 * np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))) / batch_size
    
    out = Node(loss_val, (y_pred,))
    
    def _backward():
        # Turunan BCE
        grad = (-(y_true / y_pred_clipped) + ((1 - y_true) / (1 - y_pred_clipped))) / batch_size
        y_pred.grad += grad * out.grad
        
    out._backward = _backward
    return out