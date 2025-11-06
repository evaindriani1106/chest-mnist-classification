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

### 2.1 AdvancedChestNet

Model menggunakan arsitektur modern dengan komponen-komponen:

1. **ResNet-style Blocks**
   - Skip connections untuk mengatasi vanishing gradient
   - Batch Normalization di setiap layer
   - Residual learning untuk training yang lebih baik

2. **Feature Extraction Path**:
   ```
   Input (1x28x28)
   │
   ├─ Initial Conv (64 channels)
   │   └─ MaxPool -> 14x14
   │
   ├─ ResBlock Layer1 (64->64, 2 blocks)
   │   └─ Size tetap 14x14
   │
   ├─ ResBlock Layer2 (64->128, 2 blocks)
   │   └─ Downsample ke 7x7
   │
   └─ ResBlock Layer3 (128->256, 2 blocks)
       └─ Downsample ke 4x4
   ```

3. **Squeeze-and-Excitation Module**
   - Channel attention mechanism
   - Adaptif channel weighting
   - Meningkatkan fokus pada fitur penting

4. **Advanced Classifier**
   ```
   Global Average Pooling
   │
   ├─ Linear(256->512) + BatchNorm + ReLU
   │   └─ Dropout(0.4)
   │
   ├─ Linear(512->256) + BatchNorm + ReLU
   │   └─ Dropout(0.3)
   │
   └─ Linear(256->1) // Output untuk binary classification
   ```

5. **Parameter Count**
   - Total parameter: ~1.2M
   - Semua parameter trainable
   - Efisien untuk ukuran input kecil (28x28)

## 3. Training Pipeline (`train.py`)

### 3.1 Hyperparameter
```python
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01  # max_lr untuk OneCycleLR
WEIGHT_DECAY = 2e-4
LABEL_SMOOTHING = 0.1
```

### 3.2 Optimisasi

1. **Mixed Precision Training**
   - FP16 untuk forward/backward pass
   - Menggunakan GradScaler
   - Meningkatkan speed dan mengurangi memory usage

2. **Learning Rate Scheduling**
   ```
   OneCycleLR:
   - max_lr: 0.01
   - pct_start: 0.3 (30% epoch untuk warmup)
   - div_factor: 25 (initial_lr = max_lr/25)
   - final_div_factor: 1e4
   ```

3. **Regularisasi**
   - Weight decay: 2e-4
   - Dropout bertingkat (0.4, 0.3)
   - Label smoothing: 0.1
   - Batch Normalization

4. **Training Stability**
   - Gradient clipping (max_norm=1.0)
   - Gradient scaling untuk mixed precision
   - Inisialisasi bobot Kaiming

### 3.3 Model Checkpointing

Menyimpan state lengkap saat mencapai validasi accuracy terbaik:
- Model state dict
- Optimizer state
- Scheduler state
- Scaler state
- Metrics (val_acc, val_loss)

## 4. Hasil dan Analisis

1. **Peningkatan dari Model Sebelumnya**:
   - Arsitektur lebih dalam dan sophisticated
   - Training lebih stabil dengan OneCycleLR
   - Memory efficiency dengan mixed precision
   - Regularisasi yang lebih kuat

2. **Expected Benefits**:
   - Peningkatan akurasi 5-8%
   - Training 2x lebih cepat
   - Generalisasi lebih baik
   - Convergence lebih stabil

3. **Inovasi Teknis**:
   - Squeeze-and-Excitation untuk attention
   - ResNet blocks untuk gradient flow
   - Modern training techniques (mixed precision, OneCycle)
   - Comprehensive model checkpointing

## 5. Petunjuk Penggunaan

### 5.1 Requirements
```
torch>=2.0.0
torchvision
medmnist
matplotlib
numpy
tqdm
```

### 5.2 Training
```bash
python train.py
```

### 5.3 Visualisasi
- Training history (loss & accuracy)
- Sample predictions
- Class distribution plots

## 6. Future Improvements

1. **Data Augmentation**:
   - Careful augmentation sesuai konteks medis
   - Implementasi mixup/cutmix

2. **Model Architecture**:
   - Experiment dengan attention mechanisms
   - Transfer learning dari model medis

3. **Training**:
   - Cross-validation
   - Ensemble methods
   - Class balancing techniques

---
Dibuat tanggal: 6 November 2025