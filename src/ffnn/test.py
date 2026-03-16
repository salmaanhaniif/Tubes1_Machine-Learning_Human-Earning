import numpy as np

try:
    from ffnn import FFNN
    import losses
    import utils
except ImportError:
    from .ffnn import FFNN
    from . import losses
    from . import utils

# def main():
#     print("FORWARD PROPAGATION")
    
#     # data dummy manual (tanpa utils)
#     np.random.seed(42)
#     X_dummy = np.random.randn(20, 3) # 20 sampel, 3 fitur
    
#     # Inisialisasi Model (Fitur: Xavier/He initialization dari Layer)
#     sizes = [3, 8, 3]
#     acts = ["relu", "softmax"]
    
#     model = FFNN(
#         layer_sizes=sizes, 
#         activations=acts, 
#         init_method="he", # Ngetes fitur dari layer.py
#         seed=42
#     )
    
#     print("\n[INFO] Model berhasil dibangun dengan arsitektur:", sizes)
    
#     # Tes Forward Pass (Mendapatkan objek Node)
#     output_node = model.forward(X_dummy)
#     print("\n[HASIL FORWARD PASS (Node)]")
#     print("Tipe data output :", type(output_node))
#     print("Shape output     :", output_node.data.shape)
    
#     # Tes Predict (Mendapatkan array mentah NumPy)
#     pred_array = model.predict(X_dummy)
#     print("\n[HASIL PREDICT]")
#     print(pred_array)


def main():
    print("   TESTING FINAL: FULL PIPELINE FFNN FROM SCRATCH ")

    print("--- MENYIAPKAN DATASET ---")
    np.random.seed(42)
    X = np.random.randn(300, 3)
    y = np.random.randint(0, 3, 300) 

    y_onehot = utils.create_one_hot(y, num_classes=3)
    X_train, X_val, y_train, y_val = utils.train_test_split(X, y_onehot, test_size=0.2, seed=42)
    print(f"Data Train: {X_train.shape}, Data Val: {X_val.shape}\n")

    sizes = [3, 8, 6, 3]
    acts = ["relu", "relu", "softmax"]

    model = FFNN(
        layer_sizes=sizes, 
        activations=acts, 
        learning_rate=0.05, 
        epochs=30, 
        batch_size=16, 
        l2_lambda=0.001,
        verbose=1,
        init_method="he",
        use_rms_norm=True,
        seed=42
    )

    print("\n--- DISTRIBUSI BOBOT AWAL (SEBELUM TRAINING) ---")
    model.display_weight_distribution(target_layers=[1, 2, 3])

    print("\n--- MEMULAI PELATIHAN MODEL ---")
    history = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    print("\n--- DISTRIBUSI SETELAH PELATIHAN ---")
    model.display_weight_distribution(target_layers=[1, 2, 3])
    model.display_gradient_distribution(target_layers=[1, 2, 3])

    print("\n--- MENAMPILKAN KURVA PEMBELAJARAN ---")
    utils.plot_history(history)

    print("\n--- EVALUASI PADA DATA VALIDASI ---")
    prob_preds = model.predict(X_val)
    
    y_pred_labels = np.argmax(prob_preds, axis=1)
    y_true_labels = np.argmax(y_val, axis=1)

    utils.classification_report(y_true_labels, y_pred_labels)

    print("\n--- MENGETES SAVE & LOAD ---")
    model.saveModel("final_model.pkl")
    
    loaded_model = FFNN.loadModel("final_model.pkl")
    
    new_preds = loaded_model.predict(X_val[:2])
    print("Hasil prediksi data pertama (Model Load):\n", new_preds)

if __name__ == "__main__":
    main()