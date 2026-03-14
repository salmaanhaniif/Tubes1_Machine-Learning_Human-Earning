import numpy as np
from typing import List, Tuple

"""
optimizers.py

Optimizer untuk weight update. 
Choicenya GD dan Adam
"""

class Optimizer:
    """interface yang harus diimplementasi semua optimizer."""

    def update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        grad_weights: List[np.ndarray],
        grad_biases: List[np.ndarray],
        lr: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perbarui bobot dan bias berdasarkan gradien
        
        weights      : list W per layer
        biases       : list b per layer
        grad_weights : list ∂L/∂W per layer
        grad_biases  : list ∂L/∂b per layer
        lr           : learning rate
        return       : (new_weights, new_biases)
        """


class GradientDescent(Optimizer):
    """
    Vanilla (batch) gradient descent
    """

    def update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        grad_weights: List[np.ndarray],
        grad_biases: List[np.ndarray],
        lr: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pass


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation)
    Hyperparameter default mengikuti paper: beta1=0.9, beta2=0.999, epsilon=1e-8.
      
    Referensi buat Adam: https://arxiv.org/abs/1412.6980
    """

    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """
        beta1   : decay rate moment pertama (default 0.9)
        beta2   : decay rate moment kedua   (default 0.999)
        epsilon : konstanta numerik kecil   (default 1e-8)
        """
        pass

    def _init_moments(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
        """
        Inisialisasi moment arrays (eros) saat pertama kali dipanggil.
        Dipanggil secara "lazy" di update() if belum diinisialisasi.
        """
        pass

    def update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        grad_weights: List[np.ndarray],
        grad_biases: List[np.ndarray],
        lr: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:

        pass