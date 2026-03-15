import numpy as np

try:
    from ffnn import FFNN
except ImportError:
    from .ffnn import FFNN

def main():
    print("FORWARD PROPAGATION")
    
    # data dummy manual (tanpa utils)
    np.random.seed(42)
    X_dummy = np.random.randn(20, 3) # 20 sampel, 3 fitur
    
    # Inisialisasi Model (Fitur: Xavier/He initialization dari Layer)
    sizes = [3, 8, 3]
    acts = ["relu", "softmax"]
    
    model = FFNN(
        layer_sizes=sizes, 
        activations=acts, 
        init_method="he", # Ngetes fitur dari layer.py
        seed=42
    )
    
    print("\n[INFO] Model berhasil dibangun dengan arsitektur:", sizes)
    
    # Tes Forward Pass (Mendapatkan objek Node)
    output_node = model.forward(X_dummy)
    print("\n[HASIL FORWARD PASS (Node)]")
    print("Tipe data output :", type(output_node))
    print("Shape output     :", output_node.data.shape)
    
    # Tes Predict (Mendapatkan array mentah NumPy)
    pred_array = model.predict(X_dummy)
    print("\n[HASIL PREDICT]")
    print(pred_array)

if __name__ == "__main__":
    main()