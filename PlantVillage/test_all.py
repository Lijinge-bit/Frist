"""
综合测试脚本 - 测试所有模块的基本功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim

def test_config():
    """测试配置模块"""
    print("=" * 50)
    print("测试 config 模块...")
    from config import NUM_CLASSES, DATA_DIR, IMAGE_SIZE, BATCH_SIZE, CLASS_NAMES
    print(f"  NUM_CLASSES: {NUM_CLASSES}")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  CLASS_NAMES: {len(CLASS_NAMES)} classes")
    print("✓ config 模块测试通过")
    return True

def test_models():
    """测试模型模块"""
    print("\n" + "=" * 50)
    print("测试 models 模块...")
    from models import CustomCNN, ResNetTransfer
    from config import NUM_CLASSES

    device = torch.device('cpu')

    # 测试CustomCNN
    print("  测试 CustomCNN...")
    custom_cnn = CustomCNN(num_classes=NUM_CLASSES).to(device)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = custom_cnn(dummy_input)
    assert output.shape == (2, NUM_CLASSES), f"输出形状错误: {output.shape}"
    total_params = sum(p.numel() for p in custom_cnn.parameters())
    print(f"    输出形状: {output.shape}")
    print(f"    参数数量: {total_params:,}")

    # 测试ResNetTransfer
    print("  测试 ResNetTransfer...")
    resnet = ResNetTransfer(num_classes=NUM_CLASSES).to(device)
    output = resnet(dummy_input)
    assert output.shape == (2, NUM_CLASSES), f"输出形状错误: {output.shape}"
    total_params = sum(p.numel() for p in resnet.parameters())
    trainable_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    print(f"    输出形状: {output.shape}")
    print(f"    总参数: {total_params:,}")
    print(f"    可训练参数: {trainable_params:,}")

    print("✓ models 模块测试通过")
    return True

def test_dataloader():
    """测试数据加载模块"""
    print("\n" + "=" * 50)
    print("测试 dataloader 模块...")
    from dataloader import get_data_loaders, get_transforms, TomatoDataset
    from config import DATA_DIR

    # 测试transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    print(f"  训练transform: {len(train_transform.transforms)} 个操作")
    print(f"  验证transform: {len(val_transform.transforms)} 个操作")

    # 测试数据集加载
    print("  加载数据集...")
    try:
        train_loader, val_loader, test_loader = get_data_loaders()
        print(f"  训练集: {len(train_loader.dataset)} 样本")
        print(f"  验证集: {len(val_loader.dataset)} 样本")
        print(f"  测试集: {len(test_loader.dataset)} 样本")

        # 测试一个batch
        images, labels = next(iter(train_loader))
        print(f"  Batch形状: images={images.shape}, labels={labels.shape}")

        print("✓ dataloader 模块测试通过")
        return True
    except Exception as e:
        print(f"✗ dataloader 模块测试失败: {e}")
        return False

def test_training():
    """测试训练流程"""
    print("\n" + "=" * 50)
    print("测试训练流程...")
    from models import CustomCNN
    from config import NUM_CLASSES, LEARNING_RATE, WEIGHT_DECAY

    device = torch.device('cpu')
    model = CustomCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 模拟训练
    model.train()
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    dummy_labels = torch.randint(0, NUM_CLASSES, (4,)).to(device)

    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_labels)
    loss.backward()
    optimizer.step()

    print(f"  损失值: {loss.item():.4f}")
    print("✓ 训练流程测试通过")
    return True

def test_app_imports():
    """测试Flask应用导入"""
    print("\n" + "=" * 50)
    print("测试 app 模块导入...")
    try:
        # 只测试导入，不启动服务器
        from app import GradCAM, predict, load_models
        print("  GradCAM类导入成功")
        print("  predict函数导入成功")
        print("  load_models函数导入成功")
        print("✓ app 模块导入测试通过")
        return True
    except Exception as e:
        print(f"✗ app 模块导入测试失败: {e}")
        return False

def main():
    print("番茄叶病识别系统 - 综合测试")
    print("=" * 50)

    results = []
    results.append(("config", test_config()))
    results.append(("models", test_models()))
    results.append(("dataloader", test_dataloader()))
    results.append(("training", test_training()))
    results.append(("app", test_app_imports()))

    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 50)
    if all_passed:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查错误信息。")
    print("=" * 50)

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
