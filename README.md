<br/>
<h1 align="center"> Implementasi Algoritma Pembelajaran Mesin Feed Forward Neural Network (FFNN) from Scratch dan Pengujian pada Dataset Global Student Placement and Salary</h1>

<br/>

> Tugas Besar 1 IF3270 - Pembelajaran Mesin
> By Kelompok Human earning - K01 - IF'23

<br/>

## Deskripsi Program

Repository ini berisi implementasi Feedforward Neural Network (FFNN) from scratch menggunakan Python, yang digunakan untuk memprediksi `placement_status` mahasiswa berdasarkan berbagai atribut akademik dan pengalaman kerja pada dataset Global Student Placement & Salary. Implementasi mencakup 
- forward propagation
- backward propagation dengan automatic differentiation
- berbagai fungsi aktivasi (linear, ReLU, sigmoid, tanh, softmax, swish, leaky ReLU)
- loss function (Mean Squared Error, Categorical Cross Entropy, Binary Cross Entropy)
- metode inisialisasi bobot (Zeros, Random Uniform, Random Normal, Xavier, He)
- regularisasi L1 & L2
- Optimizer SGD dan Adam.

<br/>

## Requirements
- Python ≥ 3.8
- Berbagai library python yang terdiri dari
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
    
<br/>

## Cara Instalasi dan Penggunaan
### Instalasi

1. Clone repository

``` bash   
git clone https://github.com/salmaanhaniif/Tubes1_Machine-Learning_Human-Earning.git
```

2. Install dependencies (jika belum terinstall)

``` bash   
pip install numpy pandas scikit-learn  matplotlib
```

Script utama berada di folder `src/ffnn` dan notebook berada di folder `srd`

<br/>

### Cara Penggunaan Model FFNN

``` python
from ffnn.ffnn import FFNN
from ffnn import utils

# Siapkan data
y_onehot = utils.create_one_hot(y_train, num_classes=2)

# Inisialisasi model
model = FFNN(
    layer_sizes=[11, 64, 32, 2],
    activations=["relu", "relu", "softmax"],
    init_method="he",         # zero | random_uniform | random_normal | xavier | he
    learning_rate=0.01,
    epochs=100,
    batch_size=32,
    l1_lambda=0.0,
    l2_lambda=0.001,
    optimizer="sgd",          # sgd | adam
    use_rms_norm=False,
    verbose=1,
    seed=42
)

# Training
history = model.fit(X_train, y_onehot, X_val=X_val, y_val=y_val_onehot)

# Prediksi
y_pred = model.predict(X_test)

# Visualisasi
utils.plot_history(history)
model.display_weight_distribution([1, 2, 3])
model.display_gradient_distribution([1, 2, 3])

# Save & Load
model.saveModel("model.pkl")
loaded_model = FFNN.loadModel("model.pkl")

```

Atau bisa juga untuk langsung run test.py dalam folder `src/ffnn`

``` python
python test.py
```
<br/>

## Pembagian Tugas

<table>
  <tr>
    <td> NIM </td>
    <td> Nama </td>
    <td> Tugas </td>
  </tr>
  <tr>
    <td>13523034</td>
    <td>Rafizan Muhammad Syawalazmi</td>
    <td>

* Implementasi RMS normalisasi
* Implementasi utils functions
* Implementasi visualisasi distribusi bobot, bias, dan gradien
* Membuat EDA data
* Membuat testing
* Laporan (kesimpulan, pengujian width, depth, fungsi aktivasi, inisiasi bobot, RMSNorm, dan sklearn)

    </td>
  </tr>
  <tr>
    <td>13523056</td>
    <td>Salman Hanif</td>
    <td>

* Pembuatan Struktur dasar kelas FFNN
* Implementasi Automatic Differentiation
* Implementasi Forward propagation
* Implementasi Fungsi aktivasi tambahan
* Laporan (implementasi kelas, Forward propagation) 

    </td>
  </tr>
  <tr>
    <td>13523058</td>
    <td>Noumisyifa Nabila Nareswari</td>
    <td>
* Implementasi semua loss functions
* Implementasi backward propagation
* Implementasi mekanisme save & load model
* Implementasi Adam optimizer
* Laporan (deskripsi permasalahan, backward propagation, pengujian learning rate, pengujian regularization)

    </td>
  </tr>
</table>

