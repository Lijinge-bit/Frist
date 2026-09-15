"""
快速测试脚本 - 验证基本功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Starting test...", flush=True)

import torch
print(f"PyTorch version: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

from config import NUM_CLASSES, DATA_DIR, CLASS_NAMES
print(f"NUM_CLASSES: {NUM_CLASSES}", flush=True)
print(f"DATA_DIR: {DATA_DIR}", flush=True)

from models import CustomCNN, ResNetTransfer
print("Models imported successfully", flush=True)

# 测试CustomCNN
custom_cnn = CustomCNN(num_classes=NUM_CLASSES)
dummy_input = torch.randn(1, 3, 224, 224)
output = custom_cnn(dummy_input)
print(f"CustomCNN output shape: {output.shape}", flush=True)

# 测试单样本推理（eval模式）
custom_cnn.eval()
single_input = torch.randn(1, 3, 224, 224)
single_output = custom_cnn(single_input)
print(f"CustomCNN single sample output: {single_output.shape}", flush=True)

print("All tests passed!", flush=True)
