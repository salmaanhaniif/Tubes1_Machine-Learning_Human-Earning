import numpy as np
import pickle
import matplotlib.pyplot as plt

try:
    from .layer import Layer
    from .autodiff import Node
    from . import losses
    from .optimizer import AdamOptimizer
except ImportError:
    from layer import Layer
    from autodiff import Node
    import losses
    from optimizer import AdamOptimizer

class FFNN:
    def __init__(self, layer_sizes: list, 
                 activations: list, 
                 learning_rate=0.01, 
                 epochs=100, 
                 batch_size=32, 
                 l1_lambda=0.0, 
                 l2_lambda=0.0, 
                 verbose=1,
                 init_method="random_normal", 
                 optimizer="sgd",
                 **kwargs):
        """
        layer_sizes: List jumlah neuron (contoh: [3 (input), 5 (hidden), 2 (output)])
        activations: List nama fungsi aktivasi (contoh: ["relu", "softmax"])
        """
        init_keys = {"seed", "lower_bound", "upper_bound", "mean", "std"}
        init_params = {k: v for k, v in kwargs.items() if k in init_keys}
        optimizer_params = {k: v for k, v in kwargs.items() if k not in init_keys}
        
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
        
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i], init_method, **init_params)
            self.layers.append(layer)

        if optimizer == "adam":
            self.optimizer = AdamOptimizer(
                learning_rate=optimizer_params.get("learning_rate", learning_rate),
                beta1=optimizer_params.get("beta1", 0.9),
                beta2=optimizer_params.get("beta2", 0.999),
                epsilon=optimizer_params.get("epsilon", 1e-8)
            )
        else:
            self.optimizer = None


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
        """Menggunakan self.l1_lambda dan self.l2_lambda """
        reg_loss = 0
        for layer in self.layers:
            W_data = layer.W.data
            if self.l1_lambda > 0.0:
                # penalti l1
                reg_loss += self.l1_lambda * np.sum(np.abs(W_data))
                # turunan l1
                layer.W.grad += self.l1_lambda * np.sign(W_data)
            if self.l2_lambda > 0.0:
                # penalti l2
                reg_loss += (self.l2_lambda / 2.0) * np.sum(W_data ** 2)
                # turunan l2
                layer.W.grad += self.l2_lambda * W_data
        return reg_loss
    
    def updateWeights(self):
        params = self.getParameters()
        if self.optimizer is not None:
            self.optimizer.update(params)
        else:
            for p in params:
                p.data -= self.learning_rate * p.grad
                p.grad = np.zeros_like(p.data)
        
    def _calculate_accuracy(self, y_true, y_pred_probs):
        labels_true = np.argmax(y_true, axis=1)
        labels_pred = np.argmax(y_pred_probs, axis=1)
        return np.mean(labels_true == labels_pred)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Mengurus proses pelatihan lengkap: loop epoch, batch processing, forward, loss, backward, regularisasi, update, dan pencatatan history.
        """
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        n_samples = X_train.shape[0]

        for epoch in range(self.epochs):
            # Shuffle Data di awal setiap epoch untuk memastikan model tidak belajar urutan data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train_s, y_train_s = X_train[indices], y_train[indices]

            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = int(np.ceil(n_samples / self.batch_size))

            for i in range(num_batches):
                start =  i * self.batch_size
                end = min((i + 1) * self.batch_size, n_samples)
                X_batch = X_train_s[start:end]
                y_batch = y_train_s[start:end]

                # Forward -> Loss -> Backward
                output = self.forward(X_batch)
                
                from_softmax = self.activations[-1] == 'softmax'
                loss_node = losses.categoricalCrossentropy(y_batch, output, from_softmax=from_softmax)
                
                loss_node.backward()
                
                # Regularize -> Update (Tanpa passing parameter lagi)
                reg_l = self.applyRegularization()
                self.updateWeights()

                epoch_loss += (loss_node.data + reg_l)
                epoch_acc += self._calculate_accuracy(y_batch, output.data)

            # Record Statistics
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(avg_acc)

            # Validation
            val_info = ""
            if X_val is not None and y_val is not None:
                val_out = self.predict(X_val)
                v_loss = losses.categoricalCrossentropy(y_val, Node(val_out)).data
                v_acc = self._calculate_accuracy(y_val, val_out)
                history['val_loss'].append(v_loss)
                history['val_acc'].append(v_acc)
                val_info = f" - val_loss: {v_loss:.4f} - val_acc: {v_acc:.4f}"

            if self.verbose == 1:
                progress = int((epoch + 1) / self.epochs * 20)
                bar = "=" * progress + ">" + "." * (20 - progress)
                print(f"Epoch {epoch+1:3d}/{self.epochs} [{bar}] - loss: {avg_loss:.4f} - acc: {avg_acc:.4f}{val_info}")

        return history

    # Fungsi untuk visualisasi distribusi bobot dan gradien setelah pelatihan
    # Tolong dibuat dalam format persebaran dan rata-rata karena jumlah layer biasanya sangat banyak
    

    def display_weight_distribution(self, target_layers: list):
        pass

    def display_gradient_distribution(self, target_layers: list):
        pass

    # Save & Load model
    def saveModel(self, filepath="model.pkl"):
        model_data = {
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'weights': [layer.W.data for layer in self.layers],
            'biases': [layer.b.data for layer in self.layers]
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model tersimpan: {filepath}")

    @classmethod
    def loadModel(cls, filepath="model.pkl"):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(data['layer_sizes'], data['activations'])
        for i, layer in enumerate(instance.layers):
            layer.W.data, layer.b.data = data['weights'][i], data['biases'][i]
        print(f"Model dimuat: {filepath}")
        return instance