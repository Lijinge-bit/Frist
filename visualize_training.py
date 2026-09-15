"""
训练过程可视化脚本
- 绘制训练/验证损失曲线
- 绘制训练/验证准确率曲线
- 生成模型对比图
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from config import MODEL_DIR, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, NUM_CLASSES
from dataloader import get_data_loaders
from models import CustomCNN, ResNetTransfer
from torch.optim.lr_scheduler import CosineAnnealingLR

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def train_and_record(model, model_name, train_loader, val_loader, device, epochs=10, lr=1e-3):
    """训练模型并记录过程数据"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0
    patience_counter = 0

    print(f"\n{'='*50}")
    print(f"训练 {model_name}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(scheduler.get_last_lr()[0])

        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{model_name}_best.pth'))
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"最佳验证准确率: {best_val_acc:.4f}")
    return history


def plot_training_curves(history_cnn, history_resnet, save_dir):
    """绘制训练曲线对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs_cnn = range(1, len(history_cnn['train_loss']) + 1)
    epochs_resnet = range(1, len(history_resnet['train_loss']) + 1)

    # 训练损失
    axes[0, 0].plot(epochs_cnn, history_cnn['train_loss'], 'b-o', label='CustomCNN', markersize=4)
    axes[0, 0].plot(epochs_resnet, history_resnet['train_loss'], 'r-s', label='ResNetTransfer', markersize=4)
    axes[0, 0].set_title('训练损失对比', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 验证损失
    axes[0, 1].plot(epochs_cnn, history_cnn['val_loss'], 'b-o', label='CustomCNN', markersize=4)
    axes[0, 1].plot(epochs_resnet, history_resnet['val_loss'], 'r-s', label='ResNetTransfer', markersize=4)
    axes[0, 1].set_title('验证损失对比', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 训练准确率
    axes[1, 0].plot(epochs_cnn, [a*100 for a in history_cnn['train_acc']], 'b-o', label='CustomCNN', markersize=4)
    axes[1, 0].plot(epochs_resnet, [a*100 for a in history_resnet['train_acc']], 'r-s', label='ResNetTransfer', markersize=4)
    axes[1, 0].set_title('训练准确率对比', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 验证准确率
    axes[1, 1].plot(epochs_cnn, [a*100 for a in history_cnn['val_acc']], 'b-o', label='CustomCNN', markersize=4)
    axes[1, 1].plot(epochs_resnet, [a*100 for a in history_resnet['val_acc']], 'r-s', label='ResNetTransfer', markersize=4)
    axes[1, 1].set_title('验证准确率对比', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"训练曲线已保存: {save_path}")
    return save_path


def plot_model_comparison(save_dir):
    """绘制模型对比图"""
    models = ['CustomCNN', 'ResNetTransfer']
    test_acc = [84.48, 93.92]  # 测试准确率
    params = [2889866, 24692554]  # 参数量
    train_time = [146, 222]  # 平均每epoch训练时间(秒)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 测试准确率对比
    bars1 = axes[0].bar(models, test_acc, color=['#4ECDC4', '#FF6B6B'], width=0.5)
    axes[0].set_title('测试准确率对比', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('准确率 (%)')
    axes[0].set_ylim(70, 100)
    for bar, acc in zip(bars1, test_acc):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc}%', ha='center', fontsize=11, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # 参数量对比
    bars2 = axes[1].bar(models, [p/1e6 for p in params], color=['#4ECDC4', '#FF6B6B'], width=0.5)
    axes[1].set_title('模型参数量对比', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('参数量 (百万)')
    for bar, p in zip(bars2, params):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{p/1e6:.1f}M', ha='center', fontsize=11, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # 训练时间对比
    bars3 = axes[2].bar(models, train_time, color=['#4ECDC4', '#FF6B6B'], width=0.5)
    axes[2].set_title('平均每Epoch训练时间', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('时间 (秒)')
    for bar, t in zip(bars3, train_time):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{t}s', ha='center', fontsize=11, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"模型对比图已保存: {save_path}")
    return save_path


def plot_confusion_matrix(save_dir):
    """绘制混淆矩阵热力图"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    class_names = [
        'Bacterial_spot', 'Early_blight', 'healthy', 'Late_blight',
        'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites',
        'Target_Spot', 'YellowLeaf_Curl', 'mosaic_virus'
    ]

    # ResNet的混淆矩阵数据（从训练结果）
    cm = np.array([
        [322, 2, 0, 0, 0, 4, 0, 4, 0, 0],
        [2, 139, 0, 4, 0, 1, 2, 2, 0, 0],
        [0, 0, 223, 0, 0, 0, 0, 0, 0, 0],
        [1, 15, 7, 249, 2, 4, 1, 3, 0, 1],
        [0, 0, 1, 1, 138, 5, 9, 2, 1, 5],
        [2, 3, 1, 1, 0, 265, 2, 5, 0, 1],
        [0, 0, 2, 0, 0, 0, 251, 8, 0, 1],
        [0, 1, 3, 0, 0, 0, 10, 179, 0, 0],
        [9, 0, 0, 0, 0, 0, 17, 0, 446, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 45]
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('ResNetTransfer 混淆矩阵', fontsize=14, fontweight='bold')
    ax.set_xlabel('预测类别', fontsize=11)
    ax.set_ylabel('真实类别', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")
    return save_path


if __name__ == '__main__':
    vis_dir = os.path.join(MODEL_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    print("="*60)
    print("生成训练过程可视化")
    print("="*60)

    # 加载数据
    print("\n加载数据集...")
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 训练CustomCNN
    custom_cnn = CustomCNN(num_classes=NUM_CLASSES)
    history_cnn = train_and_record(
        custom_cnn, 'custom_cnn', train_loader, val_loader, device,
        epochs=EPOCHS, lr=LEARNING_RATE
    )

    # 训练ResNetTransfer
    resnet = ResNetTransfer(num_classes=NUM_CLASSES, freeze_layers=True)
    history_resnet = train_and_record(
        resnet, 'resnet_transfer', train_loader, val_loader, device,
        epochs=EPOCHS, lr=1e-3
    )

    # 生成可视化
    print("\n生成可视化图表...")
    plot_training_curves(history_cnn, history_resnet, vis_dir)
    plot_model_comparison(vis_dir)
    plot_confusion_matrix(vis_dir)

    print("\n所有可视化图表已生成！")
    print(f"保存位置: {vis_dir}")
