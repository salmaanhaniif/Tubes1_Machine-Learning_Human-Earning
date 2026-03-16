import numpy as np

try:
    from . import activations as act
except ImportError:
    import activations as act

class Node:
    def __init__(self, data, _children=()):
        # Inisialisasi data dan gradien dalam np
        if not isinstance(data, np.ndarray):
            self.data = np.array(data, dtype=np.float64)
        else: self.data = data

        self.grad = np.zeros_like(self.data, dtype=np.float64) # init gradien dengan nilai 0 di awal
        # Fungsi mundur bawaan (kosong) dan pelacakan parent
        self._backward = lambda: None
        self._prev = set(_children)
    
    # Timpa operasi add biasa dengan operasi backward untuk grad
    def __add__(self, other):
        if not isinstance(other, Node):
            other = Node(other)
        else:
            other = other

        out = Node(self.data + other.data, (self, other))
    
        def _backward():
            # gradient untuk self
            self.grad += out.grad
            # gradient untuk other
            if self.data.shape != other.data.shape:
                other.grad += np.sum(out.grad, axis=0, keepdims=True)
            else:
                other.grad += out.grad
                
        out._backward = _backward
        return out

    # Timpa operasi matmul biasa (@) dengan operasi backward untuk grad
    def __matmul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data @ other.data, (self, other))
        
        def _backward():
            # Aturan rantai untuk perkalian matriks memerlukan Transpose (.T)
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
            
        out._backward = _backward
        return out

    def linear(self):
        out = Node(act.linear(self.data), (self,))
        def _backward():
            self.grad += act.d_linear(self.data) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Node(act.relu(self.data), (self,))
        def _backward():
            self.grad += act.d_relu(self.data) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        out = Node(act.sigmoid(self.data), (self,))
        def _backward():
            self.grad += act.d_sigmoid(self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = Node(act.tanh_fn(self.data), (self,))
        def _backward():
            self.grad += act.d_tanh(self.data) * out.grad
        out._backward = _backward
        return out

    def softmax(self):
        out = Node(act.softmax(self.data), (self,))
        def _backward():
            self.grad += act.d_softmax(self.data) * out.grad
        out._backward = _backward
        return out

    def swish(self):
        out = Node(act.swish(self.data), (self,))
        def _backward():
            self.grad += act.d_swish(self.data) * out.grad
        out._backward = _backward
        return out

    def leaky_relu(self, alpha=0.01):
        out = Node(act.leaky_relu(self.data, alpha), (self,))
        def _backward():
            self.grad += act.d_leaky_relu(self.data, alpha) * out.grad
        out._backward = _backward
        return out

    def rms_norm(self, epsilon=1e-8):
        """Root Mean Square Normalization untuk stabilitas sinyal"""
        # 1. Forward Pass
        mean_sq = np.mean(self.data ** 2, axis=1, keepdims=True)
        rms = np.sqrt(mean_sq + epsilon)
        normalized_data = self.data / rms
        out = Node(normalized_data, (self,))
        
        # 2. Backward Pass
        def _backward():
            D = self.data.shape[1] 
            sum_dy_x = np.sum(out.grad * self.data, axis=1, keepdims=True)
            term1 = out.grad / rms
            term2 = (self.data / (D * (rms ** 3))) * sum_dy_x
            self.grad += (term1 - term2)
            
        out._backward = _backward
        return out

    def backward(self):
        # Bangun urutan eksekusi mundur
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        
        # Set gradien output akhir menjadi 1 (Pemicu awal rantai kalkulus)
        self.grad = np.ones_like(self.data, dtype=np.float64)
        
        # Jalankan _backward() satu per satu dari ujung output mundur ke input
        for v in reversed(topo):
            v._backward()