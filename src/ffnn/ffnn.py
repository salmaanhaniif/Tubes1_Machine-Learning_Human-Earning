import numpy as np
import pickle
import matplotlib.pyplot as plt

try:
    from . import losses
    from .layer import Layer
    from .autodiff import Node
except ImportError:
    import losses
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
        """Menggunakan self.learning_rate yang diset di __init__"""
        params = self.getParameters()
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
    # Dibuat dalam format persebaran dan rata-rata karena jumlah layer biasanya sangat 
    
    def display_weight_distribution(self, target_layers: list):
        """
        Menampilkan statistik dasar (Min, Max, Rata-rata, Std Dev)
        dari bobot (W) dan bias (b) pada layer yang dipilih.
        """
        # Cetak judul tabel secara manual agar bentuknya langsung terlihat
        print("\n=================================================================")
        print("               DISTRIBUSI BOBOT (WEIGHTS) & BIAS                 ")
        print("=================================================================")
        print("Layer   | Param | Min        | Max        | Mean       | Std       ")
        print("-----------------------------------------------------------------")
        
        for layer_idx in target_layers:
            idx = layer_idx - 1 
            if idx < 0 or idx >= len(self.layers):
                continue
                
            # Ambil data parameter
            w_data = self.layers[idx].W.data
            b_data = self.layers[idx].b.data
            
            # Hitung Statistik Bobot (W)
            w_min = np.min(w_data)
            w_max = np.max(w_data)
            w_mean = np.mean(w_data)
            w_std = np.std(w_data)
            
            # Cetak baris Bobot (Format .4f artinya 4 angka di belakang koma, <10 artinya spasi 10 karakter)
            print(f"{layer_idx:<7} | W     | {w_min:<10.4f} | {w_max:<10.4f} | {w_mean:<10.4f} | {w_std:<10.4f}")
            
            # Hitung Statistik Bias (b)
            b_min = np.min(b_data)
            b_max = np.max(b_data)
            b_mean = np.mean(b_data)
            b_std = np.std(b_data)
            
            # Cetak baris Bias (Dikosongkan bagian nama layer agar rapi)
            print(f"        | b     | {b_min:<10.4f} | {b_max:<10.4f} | {b_mean:<10.4f} | {b_std:<10.4f}")
            print("-----------------------------------------------------------------")


    def display_gradient_distribution(self, target_layers: list):
        """
        Menampilkan statistik dasar dari sinyal error (gradien) 
        yang diterima oleh bobot dan bias saat proses backward.
        """
        print("\n=================================================================")
        print("               DISTRIBUSI GRADIEN (ERROR SIGNAL)                 ")
        print("=================================================================")
        print("Layer   | Param | Min        | Max        | Mean       | Std       ")
        print("-----------------------------------------------------------------")
        
        for layer_idx in target_layers:
            idx = layer_idx - 1 
            if idx < 0 or idx >= len(self.layers):
                continue
                
            # Ambil data gradien
            w_grad = self.layers[idx].W.grad
            b_grad = self.layers[idx].b.grad
            
            # Hitung Statistik Gradien Bobot (dW)
            dw_min = np.min(w_grad)
            dw_max = np.max(w_grad)
            dw_mean = np.mean(w_grad)
            dw_std = np.std(w_grad)
            
            print(f"{layer_idx:<7} | dW    | {dw_min:<10.4f} | {dw_max:<10.4f} | {dw_mean:<10.4f} | {dw_std:<10.4f}")
            
            # Hitung Statistik Gradien Bias (db)
            db_min = np.min(b_grad)
            db_max = np.max(b_grad)
            db_mean = np.mean(b_grad)
            db_std = np.std(b_grad)
            
            print(f"        | db    | {db_min:<10.4f} | {db_max:<10.4f} | {db_mean:<10.4f} | {db_std:<10.4f}")
            print("-----------------------------------------------------------------")

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