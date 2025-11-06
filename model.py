# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientChestNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        
        # Feature extraction blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 28x28 -> 14x14
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 14x14 -> 7x7
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 7x7 -> 3x3
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1 if num_classes == 2 else num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)     # -> 32x14x14
        x = self.conv2(x)     # -> 64x7x7
        x = self.conv3(x)     # -> 128x3x3
        
        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    BATCH_SIZE = 64
    
    print("--- Menguji Model 'EfficientChestNet' ---")
    
    # Buat model
    model = EfficientChestNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("\nArsitektur Model:")
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, 28, 28)
    output = model(dummy_input)
    
    # Hitung jumlah parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print(f"Total parameter: {total_params:,}")
    print(f"Parameter yang dapat dilatih: {trainable_params:,}")
    print("\nPengujian model 'EfficientChestNet' berhasil.")