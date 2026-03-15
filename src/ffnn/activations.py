import numpy as np
# from typing import Tuple, Callable

"""
activations.py
Berisikan fungsi aktivasi dan turunan pertamanya.

Wajib  : linear, relu, sigmoid, tanh, softmax
Bonus  : swish, leaky_relu
"""

# fungsi aktivasi

def linear(x: np.ndarray) -> np.ndarray:
    return x

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def tanh_fn(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# turunan each fungsi aktivasi

def d_linear(x: np.ndarray) -> np.ndarray:
    # Turunan dari f(x) = x adalah 1
    return np.ones_like(x)

def d_relu(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def d_sigmoid(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)

def d_tanh(x: np.ndarray) -> np.ndarray:
    # Turunan dari tanh(x) adalah 1 - tanh(x)^2
    t = tanh_fn(x)
    return 1.0 - t**2

def d_softmax(x: np.ndarray) -> np.ndarray:
    s = softmax(x)
    return s * (1.0 - s)

# Bonus 1

def swish(x: np.ndarray) -> np.ndarray:
    # Swish = x * sigmoid(x)
    return x * sigmoid(x)

def d_swish(x: np.ndarray) -> np.ndarray:
    # Turunan Swish: swish(x) + sigmoid(x) * (1 - swish(x))
    s = sigmoid(x)
    sw = x * s
    return sw + s * (1.0 - sw)

# Bonus 2

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, x * alpha)

def d_leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    # Turunan: 1 jika x > 0, alpha jika x <= 0
    dx = np.ones_like(x)
    dx[x <= 0] = alpha
    return dx


# mapping fungsi

# ACTIVATION_FUNCTIONS: dict[str, Tuple[Callable, Callable]] = {
#     "linear":     (linear,     d_linear),
#     "relu":       (relu,       d_relu),
#     "sigmoid":    (sigmoid,    d_sigmoid),
#     "tanh":       (tanh_fn,    d_tanh),
#     "softmax":    (softmax,    d_softmax),
#     "swish":      (swish,      d_swish),        
#     "leaky_relu": (leaky_relu, d_leaky_relu), 
# }

# def get_activation(name: str) -> Tuple[Callable, Callable]:
#     """
#     Kembalikan (fungsi, turunan) berdasarkan nama
#     nama yang exists: "linear", "relu", "sigmoid", "tanh", "softmax", "swish", "leaky_relu"
#     """
#     if name not in ACTIVATION_FUNCTIONS:
#         raise ValueError(f"Unknown activation function {name}, typo kali bang. Adanya: {list(ACTIVATION_FUNCTIONS)}")
#     return ACTIVATION_FUNCTIONS[name]    