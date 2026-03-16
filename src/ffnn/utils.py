import numpy as np
import matplotlib.pyplot as plt

def plot_history(history):
    """
    Menampilkan grafik Loss dan Akurasi dari hasil training.
    history: Dictionary output dari model.fit()
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    if history['val_loss'] is not None:
        plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Kurva Loss (Training vs Validation)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Akurasi
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Acc')
    if history['val_acc']:
        plt.plot(epochs, history['val_acc'], 'r-o', label='Validation Acc')
    plt.title('Kurva Akurasi (Training vs Validation)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def create_one_hot(labels, num_classes):
    """
    Mengubah label integer menjadi format One-Hot Encoding.
    Contoh: 1 -> [0, 1, 0] jika num_classes=3
    """
    return np.eye(num_classes)[labels]

def train_test_split(X, y, test_size=0.2, seed=None):
    """
    Membagi dataset menjadi data Training dan data Test secara manual.
    """
    if seed:
        np.random.seed(seed)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def classification_report(y_true, y_pred):
    """
    Menghitung metrik sederhana: Accuracy.
    y_true & y_pred: Label dalam bentuk integer (bukan one-hot).
    """
    accuracy = np.mean(y_true == y_pred)
    print(f"--- Laporan Klasifikasi ---")
    print(f"Total Data : {len(y_true)}")
    print(f"Akurasi    : {accuracy * 100:.2f}%")
    return accuracy