import numpy as np

try:
    from .autodiff import Node
except ImportError:
    from autodiff import Node

class Layer:
    def __init__(self, n_in, n_out, activation="linear", init_method="random_normal", use_rms_norm=False, **init_params):
        """
        - n_in: Jumlah neuron dari layer sebelumnya (atau jumlah fitur input).
        - n_out: Jumlah neuron di layer ini.
        - activation: Nama fungsi aktivasi (string).
        - init_method: 'zero', 'random_uniform', atau 'random_normal' 'xavier', atau 'he'.
        - init_params: Parameter tambahan untuk inisialisasi (seed, mean, variance, dll). Contoh :
            - Untuk random_uniform: lower_bound, upper_bound
            - Untuk random_normal: mean, std
            - Untuk semua metode: seed (untuk reproducibility)
            Atur aja seplenger mungkin
        """
        self.activation_name = activation.lower()
        self.use_rms_norm = use_rms_norm # Simpan pengaturan RMS Norm
        
        # Seed untuk Reproducibility
        if "seed" in init_params:
            np.random.seed(init_params["seed"])
            
        # Inisialisasi W dan b
        if init_method == "zero":
            w_data = np.zeros((n_in, n_out))
            b_data = np.zeros((1, n_out))
        elif init_method == "random_uniform":
            low = init_params.get("lower_bound", -0.1)
            high = init_params.get("upper_bound", 0.1)
            w_data = np.random.uniform(low, high, (n_in, n_out))
            b_data = np.random.uniform(low, high, (1, n_out))
        elif init_method == "random_normal":
            mean = init_params.get("mean", 0.0)
            std = init_params.get("std", 0.01)
            w_data = np.random.normal(mean, std, (n_in, n_out))
            b_data = np.random.normal(mean, std, (1, n_out))
        elif init_method == "xavier":
            std_xavier = np.sqrt(2.0 / (n_in + n_out))
            w_data = np.random.normal(0.0, std_xavier, (n_in, n_out))
            b_data = np.zeros((1, n_out))
        elif init_method == "he":
            std_he = np.sqrt(2.0 / n_in)
            w_data = np.random.normal(0.0, std_he, (n_in, n_out))
            b_data = np.zeros((1, n_out))
        else:
            raise ValueError(f"Metode inisialisasi '{init_method}' tidak dikenali.")

        # Bungkus array menjadi Node
        self.W = Node(w_data)
        self.b = Node(b_data)

    def forward(self, X):
        """
        Forward: Z = X @ W + b
        Terapkan RMS Norm (jika aktif) -> A = aktivasi(Z)
        """
        # Operasi linear
        Z = (X @ self.W) + self.b
        
        # RMS Norm
        if self.use_rms_norm:
            Z = Z.rms_norm()
        
        # Menerapkan fungsi aktivasi secara dinamis
        if self.activation_name == "linear":
            return Z.linear()
        elif self.activation_name == "relu":
            return Z.relu()
        elif self.activation_name == "sigmoid":
            return Z.sigmoid()
        elif self.activation_name == "tanh":
            return Z.tanh()
        elif self.activation_name == "softmax":
            return Z.softmax()
        elif self.activation_name == "swish":
            return Z.swish()
        elif self.activation_name == "leaky_relu":
            return Z.leaky_relu()
        else:
            raise ValueError(f"Fungsi aktivasi '{self.activation_name}' tidak tersedia, cek kembali")
            
    def getParameters(self):
        """Mengembalikan parameter (W dan b)"""
        return [self.W, self.b]