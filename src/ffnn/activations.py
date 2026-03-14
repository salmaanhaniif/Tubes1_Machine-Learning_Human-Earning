import numpy as np
from typing import Tuple, Callable

"""
activations.py
Berisikan fungsi aktivasi dan turunan pertamanya.

Wajib  : linear, relu, sigmoid, tanh, softmax
Bonus  : swish, leaky_relu
"""

# fungsi aktivasi

def linear(x: np.ndarray) -> np.ndarray:
    pass

def relu(x: np.ndarray) -> np.ndarray:
    pass

def sigmoid(x: np.ndarray) -> np.ndarray:
    pass

def tanh_fn(x: np.ndarray) -> np.ndarray:
    pass

def softmax(x: np.ndarray) -> np.ndarray:
    pass


# turunan each fungsi aktivasi

def d_linear(x: np.ndarray) -> np.ndarray:
    pass

def d_relu(x: np.ndarray) -> np.ndarray:
    pass

def d_sigmoid(x: np.ndarray) -> np.ndarray:
    pass

def d_tanh(x: np.ndarray) -> np.ndarray:
    pass

def d_softmax(x: np.ndarray) -> np.ndarray:
    pass


# Bonus 1

def swish(x: np.ndarray) -> np.ndarray:
    """Swish(x) = x * sigmoid(x)."""
    pass

def d_swish(x: np.ndarray) -> np.ndarray:
    """d/dx Swish(x) = σ(x) + x * σ(x)(1 − σ(x)) = σ(x)(1 + x(1 − σ(x)))."""
    pass


# Bonus 2

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """LeakyReLU(x) = x if x>0 else alpha*x. Default alpha=0.01."""
    pass

def d_leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """d/dx LeakyReLU(x) = 1 if x>0 else alpha."""
    pass


# mapping fungsi

ACTIVATION_FUNCTIONS: dict[str, Tuple[Callable, Callable]] = {
    "linear":     (linear,     d_linear),
    "relu":       (relu,       d_relu),
    "sigmoid":    (sigmoid,    d_sigmoid),
    "tanh":       (tanh_fn,    d_tanh),
    "softmax":    (softmax,    d_softmax),
    "swish":      (swish,      d_swish),        
    "leaky_relu": (leaky_relu, d_leaky_relu), 
}

def get_activation(name: str) -> Tuple[Callable, Callable]:
    """
    Kembalikan (fungsi, turunan) berdasarkan nama
    nama yang exists: "linear", "relu", "sigmoid", "tanh", "softmax", "swish", "leaky_relu"
    """
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function {name}, typo kali bang. Adanya: {list(ACTIVATION_FUNCTIONS)}")
    return ACTIVATION_FUNCTIONS[name]    