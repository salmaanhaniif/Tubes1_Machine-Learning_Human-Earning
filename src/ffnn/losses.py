import numpy as np
from typing import Tuple, Callable


"""
losses.py
Berisikan fungsi loss dan turunan pertamanya terhadap y (output prediksi).
Untuk turunan, yang direturn cuma turunan fungsi loss terhadap y, belum ada masalah chain rule.
Itu di backpropnya aja langsung.

"""

# fungsi loss

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error
    
    y_true : (batch,) atau (batch, 1)
    y_pred : (batch,) atau (batch, 1)
    return : float
    """
    pass

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Binary Cross-Entropy
    
    y_true : (batch,) atau (batch, 1)  - label 0 atau 1
    y_pred : (batch,) atau (batch, 1)  - probabilitas [0,1]
    return : float
    """
    pass

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Categorical Cross-Entropy
    
    y_true : (batch, C)  - yang sudah one-hot encoded
    y_pred : (batch, C)  - output softmax
    return : float
    """
    pass


# turunan fungsi loss

def d_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Turunan Mean Squared Error thd y
    
    return : same shape as y_pred
    """
    pass

def d_binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Turunan Binary Cross Entropy thd y
    
    return : same shape as y_pred
    """
    pass

def d_categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Turunan Categorical Cross Entropy thd y
    
    return : same shape as y_pred
    """
    pass


# mapping fungsi

LOSS_FUNCTIONS: dict[str, Tuple[Callable, Callable]] = {
    "mse": (mse, d_mse),
    "bce": (binary_cross_entropy, d_binary_cross_entropy),
    "cce": (categorical_cross_entropy, d_categorical_cross_entropy),
}

def get_loss(name: str) -> Tuple[Callable, Callable]:
    """
    Kembalikan (fungsi, turunan) berdasarkan nama
    nama yang exists: "mse", "bce", "cce"
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function {name}, typo kali bang. Adanya: {list(LOSS_FUNCTIONS)}")
    return LOSS_FUNCTIONS[name]    