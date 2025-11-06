# Laporan Eksperimen Klasifikasi Chest X-Ray Menggunakan Deep Learning

**Nama:** Eva Indriani  
**NIM:** 122430001  
**Program Studi:** Teknik Biomedis  
**Mata Kuliah:** Kecerdasan Buatan

## 1. Ringkasan Eksperimen

Proyek ini bertujuan mengklasifikasikan citra X-Ray dada untuk mendeteksi dua kondisi medis:
- Cardiomegaly (pembesaran jantung)
- Pneumothorax (kebocoran udara di sekitar paru-paru)

### 1.1 Dataset (`datareader.py`)

#### Sumber Data
- Dataset: ChestMNIST dari MedMNIST
- Ukuran gambar: 28x28 piksel, grayscale (1 channel)

#### Distribusi Data
```
Training Set:
- Cardiomegaly: 754 sampel
- Pneumothorax: 1552 sampel
Total: 2306 sampel

Validation Set:
- Cardiomegaly: 243 sampel
- Pneumothorax: 439 sampel
Total: 682 sampel
```

#### Preprocessing
1. **Filtering**:
   - Hanya menggunakan sampel dengan label tunggal
   - Konversi ke klasifikasi biner (Cardiomegaly=0, Pneumothorax=1)

2. **Transformasi**:
   - Konversi ke PyTorch tensor
   - Normalisasi (mean=0.5, std=0.5)
   - Data augmentation minimal untuk preservasi informasi diagnostik

## 2. Arsitektur Model (`model.py`)

### 2.1 EfficientChestNet
Model CNN yang dioptimasi untuk klasifikasi citra medis dengan fokus pada efisiensi dan kecepatan training.

#### Feature Extraction Blocks
1. **Conv Block 1** (28x28 → 14x14):
   - Conv2d(1 → 32, 3x3, padding=1)
   - BatchNorm2d
   - ReLU
   - MaxPool2d(2)

2. **Conv Block 2** (14x14 → 7x7):
   - Conv2d(32 → 64, 3x3, padding=1)
   - BatchNorm2d
   - ReLU
   - MaxPool2d(2)

3. **Conv Block 3** (7x7 → 3x3):
   - Conv2d(64 → 128, 3x3, padding=1)
   - BatchNorm2d
   - ReLU
   - MaxPool2d(2)

#### Classifier
- Dropout(0.5)
- Linear(128*3*3 → 256)
- ReLU
- Dropout(0.3)
- Linear(256 → 1)

#### Inisialisasi Bobot
- Conv layers: Kaiming initialization (fan_out, ReLU)
- BatchNorm: weights=1, bias=0
- Linear layers: Normal(0, 0.01)

#### Parameter
- Total parameter: 388,545
- Semua parameter dapat dilatih (trainable)

## 3. Training Pipeline (`train.py`)

### 3.1 Hyperparameter
```python
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
```

### 3.2 Optimisasi
- **Optimizer**: AdamW dengan weight decay
- **Loss Function**: BCEWithLogitsLoss
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Mode: min
  - Factor: 0.5
  - Patience: 3

### 3.3 Teknik Training
1. **Regularisasi**:
   - Weight decay: 1e-4
   - Dropout (0.5 dan 0.3)
   - BatchNormalization
   
2. **Stabilitas**:
   - Gradient clipping (max_norm=1.0)
   - Learning rate scheduling
   
3. **Model Checkpoint**:
   - Menyimpan model dengan validasi accuracy terbaik
   - Path: 'best_model.pth'

### 3.4 Monitoring
- Progress per batch (setiap 20 batch)
- Metrik per epoch:
  - Training loss & accuracy
  - Validation loss & accuracy
  - Learning rate saat ini
  - Best validation accuracy

## 4. Fitur-fitur Tambahan

1. **GPU Support**:
   - Otomatis deteksi dan gunakan GPU jika tersedia
   - Model dan data dipindahkan ke device yang sesuai

2. **Visualisasi**:
   - Plot history training (loss dan accuracy)
   - Visualisasi prediksi pada validation set
   - Distribusi kelas dataset

3. **Progress Tracking**:
   - Progress bar per batch
   - Summary lengkap setiap epoch
   - Best model checkpoint

## 5. Keunggulan Model

1. **Efisiensi Training**:
   - BatchNorm mempercepat konvergensi
   - Jumlah parameter optimal (388,545)
   - Dropout mencegah overfitting

2. **Arsitektur Modern**:
   - Progressive channel expansion (32→64→128)
   - MaxPooling untuk feature extraction yang lebih baik
   - Dropout ganda dengan rate berbeda
   - Kaiming initialization untuk training stabil

3. **Memory Efficient**:
   - Reduksi feature map yang progresif
   - Batch size yang lebih besar (64)
   - Optimasi memory saat training

## 6. Expected Benefits

1. **Kecepatan**:
   - Training 2-3x lebih cepat dari model sebelumnya
   - BatchNorm memungkinkan learning rate yang lebih tinggi
   - Konvergensi lebih cepat

2. **Performa**:
   - Peningkatan akurasi validasi (+3-5%)
   - Overfitting lebih rendah
   - Stabilitas training lebih baik

3. **Maintainability**:
   - Kode terstruktur dan terdokumentasi
   - Modular design
   - Mudah dimodifikasi untuk kasus lain

## 7. Catatan Penggunaan

Untuk menjalankan training:
```bash
python train.py
```

Hasil training akan menyimpan:
- Model terbaik di `best_model.pth`
- Plot history training
- Visualisasi prediksi

Pastikan semua dependencies terinstal:
```bash
pip install torch torchvision torchaudio
pip install medmnist matplotlib scikit-learn pandas tqdm
```

---
Dibuat tanggal: 6 November 2025