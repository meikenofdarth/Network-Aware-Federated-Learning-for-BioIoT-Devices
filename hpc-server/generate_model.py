import torch
import torch.nn as nn
import torch.onnx
import os

class BioNet1DCNN(nn.Module):
    def __init__(self):
        super(BioNet1DCNN, self).__init__()
        # Layer 1: Detects high-frequency local patterns in EEG
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # Layer 2: Aggregates features across temporal window
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # Global Average Pooling: Makes the model hardware-agnostic and lightweight
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [Batch, Channel, Signal_Length]
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.sigmoid(self.fc(x))

model = BioNet1DCNN()
# Dummy input representing 1 EEG data point in a 1-channel stream
dummy_input = torch.randn(1, 1, 1) 

output_path = "bio_logic.onnx"
torch.onnx.export(model, dummy_input, output_path, 
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}})

if os.path.exists(output_path):
    print(f"✅ Research Grade 1D-CNN generated: '{output_path}'")
else:
    print("❌ Error: Model generation failed.")