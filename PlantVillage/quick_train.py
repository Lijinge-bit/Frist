import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

from config import MODEL_DIR, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE
from dataloader import get_data_loaders
from models import CustomCNN, ResNetTransfer

def train_single_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / total, correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / total, correct / total

def train_model(model_name, model, train_loader, val_loader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_single_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{model_name}_best.pth'))
            print(f"  ✓ Model saved! (Best Val Acc: {best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping triggered after {epoch+1} epochs")
                break
        
        print()
    
    print(f"{model_name} training completed! Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc

def main():
    print("="*60)
    print("番茄叶病识别模型训练")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\n加载数据...")
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    print("\n训练自定义CNN模型...")
    custom_cnn = CustomCNN()
    cnn_acc = train_model('custom_cnn', custom_cnn, train_loader, val_loader, EPOCHS)
    
    print("\n训练ResNet迁移学习模型...")
    resnet_transfer = ResNetTransfer()
    resnet_acc = train_model('resnet_transfer', resnet_transfer, train_loader, val_loader, EPOCHS)
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"自定义CNN模型最佳验证准确率: {cnn_acc:.4f}")
    print(f"ResNet迁移学习模型最佳验证准确率: {resnet_acc:.4f}")
    print(f"\n模型已保存至: {MODEL_DIR}")

if __name__ == '__main__':
    main()
