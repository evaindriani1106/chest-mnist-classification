# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import AdvancedChestNet
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions
import numpy as np

# --- Hyperparameter ---
EPOCHS = 50
BATCH_SIZE = 32  # Optimal untuk model yang lebih besar
LEARNING_RATE = 0.01  # Akan di-schedule dengan OneCycleLR
WEIGHT_DECAY = 2e-4  # Sedikit lebih agresif L2 regularization
LABEL_SMOOTHING = 0.1  # Membantu generalisasi

#Menampilkan plot riwayat training dan validasi setelah training selesai.

def train():
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model
    model = AdvancedChestNet(in_channels=in_channels, num_classes=num_classes)
    print(model)
    
    # Pindahkan model ke GPU jika tersedia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nMenggunakan device: {device}")
    
    # 3. Mendefinisikan Loss Function dan Optimizer
    criterion = nn.BCEWithLogitsLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                           betas=(0.9, 0.999), eps=1e-8)
    
    # OneCycle learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # 30% waktu untuk warmup
        div_factor=25,  # initial_lr = max_lr/25
        final_div_factor=1e4,  # final_lr = initial_lr/1e4
    )
    
    # Gradient Scaler untuk mixed precision training
    scaler = GradScaler()
    
    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    best_val_acc = 0.0
    print("\n--- Memulai Training ---")
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().to(device)
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Gradient scaling backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaling
            scaler.step(optimizer)
            scaler.update()
            
            # Learning rate step
            scheduler.step()
            
            # Update metrics
            running_loss += loss.item()
            with torch.no_grad():
                predicted = (outputs > 0).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Progress update setiap 10 batch
            if (batch_idx + 1) % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f'Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} LR: {current_lr:.6f}')
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Simpan model terbaik dengan validasi metrics
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_acc': val_accuracy,
                'val_loss': avg_val_loss,
            }, 'best_model.pth')
        
            # Simpan history dan print progress
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)

    print("\n=== Training Selesai ===")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Load model terbaik untuk evaluasi
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Tampilkan plot
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi pada validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    train()
    