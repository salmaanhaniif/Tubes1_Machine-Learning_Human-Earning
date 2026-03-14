import numpy as np
from typing import Optional, Tuple

"""
normalization.py
Implementasi RMSNorm sebagai layer normalisasi.

RMSNorm (Root Mean Square Layer Normalization):
  RMS(x) = sqrt( (1/n) Σ x_i² )
  y = (x / RMS(x)) * gain

Di mana gain adalah parameter yang dapat dipelajari, diinisialisasi dengan ones.

Referensi: https://arxiv.org/abs/1910.07467
"""

class RMSNorm:
    """
    RMSNorm layer
    
    Pakenya disisipin antara linear transform dan fungsi aktivasi.

    Cara pake di FFNN:
      z = X @ W + b          
      z_norm = rms.forward(z) <- here, before aktivasi
      a = activation(z_norm)
    """

    def __init__(self, n_features: int, epsilon: float = 1e-8) -> None:
        """
        n_features : jumlah fitur / ukuran dimensi terakhir input
        epsilon    : konstanta kecil untuk stabilitas numerik
        """
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass RMSNorm
        
        x      : (batch, n_features)
        return : (batch, n_features) 
        
        Cache intermediate values untuk backward.
        """
        pass

    def backward(self, d_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass, bakal hitung gradien terhadap input dan gain.
        d_out    : (batch, n_features) - gradien dari layer atas
        return   : (d_x, d_gamma)
          d_x     : (batch, n_features) - gradien ke layer bawah
          d_gamma : (n_features,)       - gradien untuk update gain
        """
        pass

    def update(self, d_gamma: np.ndarray, lr: float) -> None:
        """
        Update parameter gain dengan gradient descent.
        d_gamma : gradien dari backward()
        lr      : learning rate
        """
        pass