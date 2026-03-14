import numpy as np
from typing import Tuple, Optional, Callable

"""
initializers.py
Berisikian fungsi-fungsi untuk all types inisialisasi bobot dan bias.
Setiap fungsi terima shape tuple dan return np.ndarray.
"""

# berbagai fungsi inisialisasi bobot

def zeros(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Zero initialization: init semua elemen = 0
    
    shape  : tuple dimensi array, misal (n_in, n_out)
    return : np.ndarray of zeros
    """
    pass

def uniform(
    shape: Tuple[int, ...],
    low: float = -0.1,
    high: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Random uniform dalam [low, high]
    
    shape  : tuple dimensi
    low    : batas bawah
    high   : batas atas
    seed   : random seed untuk reproducibility (None = tidak di set)
    return : np.ndarray
    """
    pass

def normal(
    shape: Tuple[int, ...],
    mean: float = 0.0,
    variance: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Random normal dengan mean dan variance tertentu
    
    shape    : tuple dimensi
    mean     : rata-rata distribusi
    variance : variansi (bukan std)
    seed     : random seed
    return   : np.ndarray
    """
    pass


# bonus

def xavier_uniform(
    shape: Tuple[int, ...],
    fan_in: int,
    fan_out: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Xavier/Glorot uniform
    
    limit = sqrt(6 / (fan_in + fan_out))
    distribusi U(-limit, +limit)
    """
    pass

def xavier_normal(
    shape: Tuple[int, ...],
    fan_in: int,
    fan_out: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Xavier/Glorot normal
    
    std = sqrt(2 / (fan_in + fan_out))
    """
    pass

def he_uniform(
    shape: Tuple[int, ...],
    fan_in: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    He uniform
    
    limit = sqrt(6 / fan_in)
    """
    pass

def he_normal(
    shape: Tuple[int, ...],
    fan_in: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    He normal
    
    std = sqrt(2 / fan_in)
    """
    pass


# mapping

INITIALIZER_FUNCTIONS: dict[str, Callable] = {
    "zeros":          zeros,
    "uniform":        uniform,
    "normal":         normal,
    "xavier_uniform": xavier_uniform,   
    "xavier_normal":  xavier_normal,    
    "he_uniform":     he_uniform,       
    "he_normal":      he_normal,        
}

def get_initializer(name: str) -> Callable:
    """
    Kembalikan fungsi init berdasarkan nama
    nama yang exists: liat aja keys di dict INITIALIZER_FUNCTIONS
    """
    if name not in INITIALIZER_FUNCTIONS:
        raise ValueError(f"Unknown initializer function {name}, typo kali bang. Adanya: {list(INITIALIZER_FUNCTIONS)}")
    return INITIALIZER_FUNCTIONS[name]    