"""
模型结构可视化脚本
- 生成模型结构图
- 统计模型参数
"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import rcParams

from config import MODEL_DIR, NUM_CLASSES
from models import CustomCNN, ResNetTransfer

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def generate_model_structure(model, model_name, save_dir):
    """生成模型结构图（文本版本）"""
    try:
        # 使用batch_size=2避免BatchNorm问题
        device = next(model.parameters()).device
        dummy_input = torch.randn(2, 3, 128, 128).to(device)

        # 收集模型结构信息
        structure_lines = []
        structure_lines.append(f"{'='*60}")
        structure_lines.append(f"{model_name} 模型结构")
        structure_lines.append(f"{'='*60}")

        # 统计各层信息
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d,
                                  nn.AdaptiveAvgPool2d, nn.Linear, nn.Dropout)):
                params = sum(p.numel() for p in module.parameters())
                if isinstance(module, nn.Conv2d):
                    structure_lines.append(f"Conv2d({module.in_channels}, {module.out_channels}, "
                                         f"kernel={module.kernel_size}, stride={module.stride}) - {params:,} params")
                elif isinstance(module, nn.Linear):
                    structure_lines.append(f"Linear({module.in_features}, {module.out_features}) - {params:,} params")
                elif isinstance(module, nn.BatchNorm2d):
                    structure_lines.append(f"BatchNorm2d({module.num_features}) - {params:,} params")
                elif isinstance(module, nn.Dropout):
                    structure_lines.append(f"Dropout(p={module.p})")
                elif isinstance(module, nn.MaxPool2d):
                    structure_lines.append(f"MaxPool2d(kernel={module.kernel_size}, stride={module.stride})")
                elif isinstance(module, nn.AdaptiveAvgPool2d):
                    structure_lines.append(f"AdaptiveAvgPool2d({module.output_size})")

        # 保存为文本文件
        save_path = os.path.join(save_dir, f'{model_name}_structure.txt')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(structure_lines))
        print(f"模型结构已保存: {save_path}")

        # 同时打印到控制台
        print('\n'.join(structure_lines[:20]) + '\n...')

        return save_path
    except Exception as e:
        print(f"生成{model_name}结构失败: {e}")
        return None


def count_parameters(model):
    """统计模型参数"""
    total_params = 0
    trainable_params = 0
    layer_params = []

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            layer_params.append((name, num_params, True))
        else:
            layer_params.append((name, num_params, False))

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'layers': layer_params
    }


def plot_parameter_distribution(model, model_name, save_dir):
    """绘制参数分布图"""
    stats = count_parameters(model)

    # 按模块统计参数
    module_params = {}
    for name, params, trainable in stats['layers']:
        module = name.split('.')[0] if '.' in name else name
        if module not in module_params:
            module_params[module] = {'trainable': 0, 'frozen': 0}
        if trainable:
            module_params[module]['trainable'] += params
        else:
            module_params[module]['frozen'] += params

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 饼图：可训练 vs 冻结参数
    sizes = [stats['trainable'], stats['frozen']]
    labels = [f"可训练\n{stats['trainable']/1e6:.1f}M", f"冻结\n{stats['frozen']/1e6:.1f}M"]
    colors = ['#4ECDC4', '#FF6B6B']
    explode = (0.05, 0)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title(f'{model_name} 参数分布', fontsize=12, fontweight='bold')

    # 柱状图：各模块参数量
    modules = list(module_params.keys())[:10]  # 只显示前10个模块
    trainable_counts = [module_params[m]['trainable']/1e6 for m in modules]
    frozen_counts = [module_params[m]['frozen']/1e6 for m in modules]

    x = range(len(modules))
    width = 0.35

    ax2.bar(x, trainable_counts, width, label='可训练', color='#4ECDC4')
    ax2.bar([i + width for i in x], frozen_counts, width, label='冻结', color='#FF6B6B')

    ax2.set_title('各模块参数量', fontsize=12, fontweight='bold')
    ax2.set_xlabel('模块')
    ax2.set_ylabel('参数量 (百万)')
    ax2.set_xticks([i + width/2 for i in x])
    ax2.set_xticklabels(modules, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_params.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"参数分布图已保存: {save_path}")
    return save_path


if __name__ == '__main__':
    vis_dir = os.path.join(MODEL_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    print("="*60)
    print("模型结构可视化")
    print("="*60)

    device = torch.device('cpu')

    # CustomCNN
    print("\n分析 CustomCNN...")
    custom_cnn = CustomCNN(num_classes=NUM_CLASSES).to(device)
    cnn_stats = count_parameters(custom_cnn)
    print(f"  总参数: {cnn_stats['total']:,}")
    print(f"  可训练参数: {cnn_stats['trainable']:,}")
    print(f"  冻结参数: {cnn_stats['frozen']:,}")
    plot_parameter_distribution(custom_cnn, 'CustomCNN', vis_dir)
    generate_model_structure(custom_cnn, 'CustomCNN', vis_dir)

    # ResNetTransfer
    print("\n分析 ResNetTransfer...")
    resnet = ResNetTransfer(num_classes=NUM_CLASSES, freeze_layers=True).to(device)
    resnet_stats = count_parameters(resnet)
    print(f"  总参数: {resnet_stats['total']:,}")
    print(f"  可训练参数: {resnet_stats['trainable']:,}")
    print(f"  冻结参数: {resnet_stats['frozen']:,}")
    plot_parameter_distribution(resnet, 'ResNetTransfer', vis_dir)
    generate_model_structure(resnet, 'ResNetTransfer', vis_dir)

    print("\n所有可视化图表已生成！")
    print(f"保存位置: {vis_dir}")
