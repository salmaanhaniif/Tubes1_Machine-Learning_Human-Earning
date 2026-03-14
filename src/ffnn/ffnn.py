import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple, Any

from .activations import get_activation
from .losses import get_loss
from .initializers import get_initializer
from .regularization import l1_grad, l2_grad, l1_penalty, l2_penalty
from .optimizers import GradientDescent, Adam, Optimizer
from .normalization import RMSNorm

"""
ffnn.py
Kelas utama Feedforward Neural Network (FFNN) 

Cara pake:

model = FFNN(layer_sizes=[11, 64, 32, 1],
            activations=['relu', 'relu', 'sigmoid'],
            loss='bce')
history = model.fit(X_train, y_train, X_val, y_val,
                    batch_size=32, lr=0.01, epochs=100, verbose=1)
y_pred  = model.predict(X_test)

"""


class FFNN:
    """
    Feedforward Neural Network

    Atribut publik
    --------------
    layer_sizes   : list[int]        - ukuran tiap layer [input, ..., output]
    activations   : list[str]        - nama aktivasi per layer (len = n_layers - 1)
    loss_name     : str              - nama loss function
    weights       : list[np.ndarray] - W per layer, shape (n_in, n_out)
    biases        : list[np.ndarray] - b per layer, shape (1, n_out)
    grad_weights  : list[np.ndarray] - ∂L/∂W per layer (diisi setelah backward)
    grad_biases   : list[np.ndarray] - ∂L/∂b per layer

    Atribut privat
    --------------
    _z_cache : list[np.ndarray] - pre-activation z tiap layer
    _a_cache : list[np.ndarray] - post-activation a; index 0 = input X
     _norm_layers  : list[RMSNorm|None] RMSNorm per layer (None jika tidak dipakai)
    _optimizer    : Optimizer - instance GradientDescent atau Adam
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        loss: str,
        weight_init: str = "uniform",
        init_params: Optional[Dict[str, Any]] = None,
        regularization: Optional[str] = None,
        lambda_: float = 0.0,
        optimizer: str = "sgd",
        optimizer_params: Optional[Dict[str, Any]] = None,
        use_rmsnorm: bool = False,
    ) -> None:
        """
        Params
        ------
        layer_sizes      : [input_dim, hidden1, ..., output_dim]
        activations      : nama aktivasi per layer, len = len(layer_sizes) - 1
                           contoh: ['relu', 'relu', 'sigmoid']
        loss             : 'mse' | 'bce' | 'cce'
        weight_init      : 'zeros' | 'uniform' | 'normal' |
                           'xavier_uniform' | 'xavier_normal' |
                           'he_uniform' | 'he_normal'
        init_params      : dict buat kebutuhan initializer
                           uniform -> {'low': -0.1, 'high': 0.1, 'seed': 42}
                           normal  -> {'mean': 0.0, 'variance': 0.1, 'seed': 42}
        regularization   : None | 'l1' | 'l2'
        lambda_          : koef regularisasi
        optimizer        : 'sgd' | 'adam'
        optimizer_params : dict params untuk Adam, e.g. {'beta1': 0.9, ...}
        use_rmsnorm      : True | False (mau pakai rmsnorm atau tidak)
        """
        pass

    # inits

    def _init_weights(self) -> None:
        """
        Inisialisasi self.weights dan self.biases untuk semua layer.
        Dipanggil dari __init__.
        - W shape : (layer_sizes[l], layer_sizes[l+1])
        - b shape : (1, layer_sizes[l+1])
        - grad_weights dan grad_biases diinisialisasi zeros, shape sama.
        """
        pass

    def _init_norm_layers(self) -> None:
        """
        Buat list self._norm_layers
        
        If use_rmsnorm=True : tiap elemen = RMSNorm(layer_sizes[l+1])
        If use_rmsnorm=False: tiap elemen = None
        
        Output layer tidak dinormalisasi (selalu None di index terakhir)
        """
        pass

    def _init_optimizer(self) -> None:
        """
        Buat instance optimizer (GradientDescent atau Adam) dan simpan
        ke self._optimizer.
        """
        pass

    # forward prop

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass melalui semua layer

        X      : (batch, input_dim)
        return : (batch, output_dim) - output prediksi y
        
        Alur per layer l:
          z = a_prev @ W[l] + b[l]
          z = norm_layers[l].forward(z)   <- only if use_rmsnorm dan bukan output layer
          a = activation_l(z)

        Side effect: 
          _a_cache[0]   = X
          _z_cache[l]   = z sebelum aktivasi (sesudah RMSNorm jika aktif)
          _a_cache[l+1] = a
        """
        pass

    # backward prop

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Backward pass
        Hitung gradien untuk semua bobot dan bias.
        Hasil disimpan ke self.grad_weights dan self.grad_biases.

        X : (batch, input_dim)
        y : (batch, output_dim)

        Algoritma (chain rule, iterasi mundur):
          1. dL_dy_hat = d_loss(y, y_hat)
          2. Untuk tiap layer l mundur:
             a. Jika softmax + cce -> dL_dz = (y_hat - y) / batch  (intinya gt)
                Selain itu         -> dL_dz = dL_dy_hat * d_activation(z_cache[l])
             b. Jika use_rmsnorm   -> (dL_dz, d_gamma) = norm_layers[l].backward(dL_dz)
                                     simpan d_gamma untuk update norm layer
             c. grad_W[l] = a_cache[l].T @ dL_dz
             d. grad_b[l] = mean(dL_dz, axis=0, keepdims=True)
             e. dL_dy_hat (untuk layer l-1) = dL_dz @ W[l].T
          3. Tambahkan regularisasi grad ke grad_W[l] (bukan grad_b)
        """
        pass

    # update weight

    def update_weights(self, lr: float) -> None:
        """
        Update bobot menggunakan optimizer.
        Regularisasi sudah ditambahkan ke gradien di backward().
        lr : learning rate
        """
        pass

    # fitting dan training loop

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32,
        lr: float = 0.01,
        epochs: int = 100,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Latih model dengan mini-batch gradient descent.

        Parameters
        ----------
        X_train, y_train  : training sata
        X_val, y_val      : data validasi (opsional; val_loss = None jika tidak ada)
        batch_size        : ukuran mini-batch
        lr                : learning rate
        epochs            : jumlah epoch
        verbose           : 0=silent, 1=progress bar + loss tiap epoch

        Returns
        -------
        history : {'train_loss': [...], 'val_loss': [...]}  panjang = epochs
                  val_loss berisi None tiap epoch if X_val tidak diberikan
        """
        pass

    def _run_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        lr: float,
    ) -> float:
        """
        Jalankan satu epoch: iterasi mini-batch, backward, update.
        return : training loss rata-rata epoch ini
        """
        pass

    # predict

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediksi output untuk data baru
        
        X      : (n_samples, input_dim)
        return : (n_samples, output_dim)
        """
        pass

    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Kembalikan prediksi label kelas
        
        - output_dim == 1 : threshold pada probs sigmoid
        - output_dim > 1  : argmax (multiclass softmax)
        return : (n,) int
        """
        pass

    # loss evaluation

    def compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        include_regularization: bool = False,
    ) -> float:
        """
        Hitung loss (opsional: termasuk penalty regularisasi).
        y_true                 : aktual
        y_pred                 : hasil prediksi
        include_regularization : tambahkan L1/L2 penalty ke nilai loss
        return                 : float
        """
        pass

    # visualisasi distribusi

    def plot_weight_dist(
        self,
        layer_indices: Optional[List[int]] = None,
        bins: int = 50,
        figsize: Tuple[int, int] = (12, 4),
    ) -> None:
        """
        Plot histogram distribusi bobot per layer.

        layer_indices : list index layer yang akan di-plot
                        None -> plot semua layer
                        Contoh: [0, 2] -> hanya layer 0 dan 2
        bins          : jumlah bin histogram
        figsize       : ukuran figure matplotlib
        """
        pass

    def plot_grad_dist(
        self,
        layer_indices: Optional[List[int]] = None,
        bins: int = 50,
        figsize: Tuple[int, int] = (12, 4),
    ) -> None:
        """
        Plot histogram distribusi gradien bobot per layer.
        Harus dipanggil setelah backward() pernah dijalankan.

        layer_indices : list index layer. If None = semua layer
        """
        pass

    # save & load

    def save(self, filepath: str) -> None:
        """
        Simpan model ke file .npz (bobot) + .json (config)
        
        filepath : path tanpa ekstensi
            contoh: 'model/ffnn_v1', akan membuat
                -> 'model/ffnn_v1_weights.npz' (W, b, dan gain jika rmsnorm)
                -> 'model/ffnn_v1_config.json' (hyperparameter & arsitektur)
        """
        pass

    @classmethod
    def load(cls, filepath: str) -> "FFNN":
        """
        Load model dari file yang disimpan oleh save()
        
        filepath : path tanpa ekstensi
        return   : instance FFNN dengan bobot restored
        """
        pass

    # gk diminta spek but buat print printan

    def summary(self) -> None:
        """
        Tampilkan ringkasan arsitektur model ke stdout:
        layer index, input size, output size, activation, jumlah parameter.
        """
        pass

    def __repr__(self) -> str:
        """
        Contohnya bakal keprint:
        FFNN([11, 64, 32, 1], activations=['relu','relu','sigmoid'], loss='bce')
        """
        
        pass