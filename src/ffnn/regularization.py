import numpy as np
from typing import List

"""
regularization.py
Berisikan fungsi regularisasi, which ada L1 dan L2.

Turunannya dipake di FFNN.update_weights() buat nambah gradien sebelum
langkah gradient descent:
    grad_total = grad_loss + grad_regularization
"""

# L1

def l1_penalty(weights: List[np.ndarray], lambda_: float) -> float:
    """
    Hitung total L1 penalty untuk semua matriks bobot
    
    weights  : list of W arrays (tidak termasuk bias)
    lambda_  : koefisien regularisasi
    return   : float
    """
    pass

def l1_grad(W: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Gradien L1 terhadap satu matriks bobot W
    
    W       : np.ndarray - bobot satu layer
    lambda_ : koefisien regularisasi
    return  : np.ndarray same shape as W
    """
    pass


# L2

def l2_penalty(weights: List[np.ndarray], lambda_: float) -> float:
    """
    Hitung total L2 penalty untuk semua matriks bobot
    
    weights  : list of W arrays
    lambda_  : koefisien regularisasi
    return   : float
    """
    pass

def l2_grad(W: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Gradien L2 terhadap satu matriks bobot W

    W       : np.ndarray
    lambda_ : koefisien regularisasi
    return  : np.ndarray same shape as W
    """
    pass
