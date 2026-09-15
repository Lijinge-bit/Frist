import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import time

from config import *
from dataloader import get_data_loaders
from models import CustomCNN, ResNetTransfer

class EarlyStopping:
    """早停机制，支持基于loss或accuracy的监控"""
    def __init__(self, patience=PATIENCE, verbose=False, delta=0, mode='loss'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode  # 'loss' 或 'accuracy'

        if mode == 'loss':
            self.best_value = np.inf
        else:
            self.best_value = -np.inf

    def __call__(self, val_value, model, model_name):
        if self.mode == 'loss':
            score = -val_value
            improved = val_value < self.best_value - self.delta
        else:
            score = val_value
            improved = val_value > self.best_value + self.delta

        if self.best_score is None:
            self.best_score = score
            self.best_value = val_value
            self.save_checkpoint(model, model_name)
        elif improved:
            self.best_score = score
            self.best_value = val_value
            self.save_checkpoint(model, model_name)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, model_name):
        if self.verbose:
            if self.mode == 'loss':
                print(f'Validation loss decreased to {self.best_value:.6f}. Saving model...')
            else:
                print(f'Validation accuracy increased to {self.best_value:.4f}. Saving model...')
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{model_name}_best.pth'))

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, model_name, num_epochs=EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        start_time = time.time()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        scheduler.step()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        epoch_time = time.time() - start_time

        print(f'Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 使用验证准确率作为早停指标
        early_stopping(val_acc, model, model_name)

        if early_stopping.early_stop:
            print('Early stopping triggered.')
            break

    return train_losses, val_losses, train_accs, val_accs

def test_model(model, test_loader, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model_path = os.path.join(MODEL_DIR, f'{model_name}_best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f'\n{model_name} Test Results:')
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))
    print('\nConfusion Matrix:')
    print(confusion_matrix(all_labels, all_preds))

    test_acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
    print(f'\nTest Accuracy: {test_acc:.4f}')

    return test_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, val_loader, test_loader = get_data_loaders()
    print(f'Data loaded: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}')

    # 训练CustomCNN
    print('\n=== Training CustomCNN ===')
    custom_cnn = CustomCNN().to(device)
    criterion_cnn = nn.CrossEntropyLoss()
    optimizer_cnn = optim.AdamW(custom_cnn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler_cnn = CosineAnnealingLR(optimizer_cnn, T_max=EPOCHS)
    early_stopping_cnn = EarlyStopping(patience=PATIENCE, verbose=True, mode='accuracy')

    train_model(custom_cnn, train_loader, val_loader, criterion_cnn, optimizer_cnn, scheduler_cnn, early_stopping_cnn, 'custom_cnn')
    test_model(custom_cnn, test_loader, 'custom_cnn')

    # 训练ResNetTransfer（使用分层学习率）
    print('\n=== Training ResNetTransfer ==='
    resnet_transfer = ResNetTransfer(freeze_layers=True).to(device)
    criterion_resnet = nn.CrossEntropyLoss()

    # 使用分层学习率：预训练层用小学习率，分类头用大学习率
    param_groups = resnet_transfer.get_param_groups(lr_pretrained=1e-5, lr_classifier=1e-3)
    optimizer_resnet = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler_resnet = CosineAnnealingLR(optimizer_resnet, T_max=EPOCHS)
    early_stopping_resnet = EarlyStopping(patience=PATIENCE, verbose=True, mode='accuracy')

    # 打印可训练参数信息
    total_params = sum(p.numel() for p in resnet_transfer.parameters())
    trainable_params = sum(p.numel() for p in resnet_transfer.parameters() if p.requires_grad)
    print(f'ResNet 总参数: {total_params:,}')
    print(f'ResNet 可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)')

    train_model(resnet_transfer, train_loader, val_loader, criterion_resnet, optimizer_resnet, scheduler_resnet, early_stopping_resnet, 'resnet_transfer')
    test_model(resnet_transfer, test_loader, 'resnet_transfer')

if __name__ == '__main__':
    main()