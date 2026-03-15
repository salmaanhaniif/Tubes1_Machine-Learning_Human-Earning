import numpy as np
import pickle
import matplotlib.pyplot as plt

try:
    from .layer import Layer
    from .autodiff import Node
except ImportError:
    from layer import Layer
    from autodiff import Node

class FFNN:
    def __init__(self, layer_sizes: list, 
                 activations: list, 
                 learning_rate=0.01, 
                 epochs=10, 
                 batch_size=32, 
                 l1_lambda=0.0, 
                 l2_lambda=0.0, 
                 verbose=1,
                 init_method="random_normal", 
                 **init_params):
        """
        layer_sizes: List jumlah neuron (contoh: [3 (input), 5 (hidden), 2 (output)])
        activations: List nama fungsi aktivasi (contoh: ["relu", "softmax"])
        """
        if len(layer_sizes) - 1 != len(activations):
            raise ValueError("Jumlah fungsi aktivasi tidak sama dengan jumlah transisi layer, cek kembali")
        
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.layers = []

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.verbose = verbose
        
        for i in range(0, len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]
            act_func = activations[i]
            
            layer = Layer(n_in, n_out, act_func, init_method, **init_params)
            self.layers.append(layer)


    def forward(self, X):
        """Proses forward yang menghasilkan Node"""
        out = X if isinstance(X, Node) else Node(X)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, X):
        """Proses forward yang mengembalikan array mentah (untuk inference)"""
        output_node = self.forward(X)
        return output_node.data

    def getParameters(self):
        """Mengambil semua objek Node (W dan b) dari seluruh layer untuk keperluan update"""
        params = []
        for layer in self.layers:
            params.extend(layer.getParameters())
        return params

    def applyRegularization(self):
        pass
    
    def updateWeights(self):
        pass

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        pass

    # Fungsi untuk visualisasi distribusi bobot dan gradien setelah pelatihan
    # Tolong dibuat dalam format persebaran dan rata-rata karena jumlah layer biasanya sangat banyak
    
    def display_weight_distribution(self, target_layers: list):
        pass

    def display_gradient_distribution(self, target_layers: list):
        pass

    # Save & Load model
    def saveModel(self, filepath="model.pkl"):
        pass

    @classmethod
    def loadModel(cls, filepath="model.pkl"):
        pass