"""
错误案例分析脚本
- 找出被错误分类的样本
- 保存错误案例图片
- 生成错误分析报告
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image
from torchvision import transforms

from config import MODEL_DIR, CLASS_NAMES, IMAGE_SIZE, MEAN, STD, DATA_DIR
from models import ResNetTransfer
from dataloader import get_data_loaders

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 类别中文名
CLASS_NAMES_CN = {
    'Tomato_Bacterial_spot': '细菌性斑点病',
    'Tomato_Early_blight': '早疫病',
    'Tomato_healthy': '健康叶片',
    'Tomato_Late_blight': '晚疫病',
    'Tomato_Leaf_Mold': '叶霉病',
    'Tomato_Septoria_leaf_spot': '斑枯病',
    'Tomato_Spider_mites_Two_spotted_spider_mite': '蜘蛛螨',
    'Tomato__Target_Spot': '靶斑病',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': '黄曲叶病毒',
    'Tomato__Tomato_mosaic_virus': '花叶病毒'
}


def find_error_cases(model, test_loader, device, num_cases=10):
    """找出错误分类的样本"""
    model.eval()
    model = model.to(device)

    error_cases = []

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            # 找出错误预测
            wrong_mask = preds != labels
            if wrong_mask.any():
                wrong_indices = wrong_mask.nonzero().squeeze()
                if wrong_indices.dim() == 0:
                    wrong_indices = wrong_indices.unsqueeze(0)

                for idx in wrong_indices:
                    true_label = labels[idx].item()
                    pred_label = preds[idx].item()
                    confidence = probs[idx][pred_label].item()

                    error_cases.append({
                        'image': images[idx].cpu(),
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': confidence,
                        'true_class': CLASS_NAMES[true_label],
                        'pred_class': CLASS_NAMES[pred_label]
                    })

                    if len(error_cases) >= num_cases:
                        return error_cases

    return error_cases


def visualize_error_cases(error_cases, save_dir):
    """可视化错误案例"""
    if not error_cases:
        print("没有找到错误案例")
        return

    # 按错误类型分组
    error_types = {}
    for case in error_cases:
        key = f"{CLASS_NAMES_CN[case['true_class']]} → {CLASS_NAMES_CN[case['pred_class']]}"
        if key not in error_types:
            error_types[key] = []
        error_types[key].append(case)

    # 统计错误类型
    print("\n" + "="*60)
    print("错误案例统计")
    print("="*60)
    for error_type, cases in sorted(error_types.items(), key=lambda x: -len(x[1])):
        print(f"{error_type}: {len(cases)} 例")

    # 绘制错误案例图片
    num_show = min(len(error_cases), 12)
    cols = 4
    rows = (num_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < num_show:
            case = error_cases[i]
            img = case['image']

            # 反归一化显示
            img = img * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()

            ax.imshow(img)
            true_cn = CLASS_NAMES_CN[case['true_class']]
            pred_cn = CLASS_NAMES_CN[case['pred_class']]
            ax.set_title(f"真实: {true_cn}\n预测: {pred_cn}\n置信度: {case['confidence']:.2%}",
                        fontsize=9, color='red')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle('错误案例展示', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'error_cases.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n错误案例图片已保存: {save_path}")

    return save_path


def generate_error_report(error_cases, save_dir):
    """生成错误分析报告"""
    report = []
    report.append("="*60)
    report.append("错误案例分析报告")
    report.append("="*60)

    # 统计
    total_errors = len(error_cases)
    error_types = {}
    for case in error_cases:
        key = (case['true_class'], case['pred_class'])
        if key not in error_types:
            error_types[key] = 0
        error_types[key] += 1

    report.append(f"\n总错误数: {total_errors}")
    report.append(f"错误类型数: {len(error_types)}")

    report.append("\n常见错误类型:")
    report.append("-"*60)
    for (true_cls, pred_cls), count in sorted(error_types.items(), key=lambda x: -x[1])[:10]:
        true_cn = CLASS_NAMES_CN[true_cls]
        pred_cn = CLASS_NAMES_CN[pred_cls]
        report.append(f"  {true_cn} → {pred_cn}: {count} 例")

    # 分析
    report.append("\n" + "="*60)
    report.append("错误原因分析")
    report.append("="*60)

    report.append("""
1. 视觉相似性:
   - 晚疫病和早疫病症状相似，都表现为叶片枯萎变褐
   - 蜘蛛螨和靶斑病都会在叶片上形成斑点
   - 叶霉病和部分病害的叶片变色相似

2. 病害阶段影响:
   - 早期病害特征不明显，难以准确识别
   - 多种病害可能同时存在，混淆判断

3. 图像质量因素:
   - 拍摄角度、光照条件影响特征提取
   - 图片分辨率限制，细微差异难以区分

4. 类别不平衡:
   - 样本较少的类别（如花叶病毒）识别难度较大
   - 模型对少数类别的学习不充分
""")

    # 保存报告
    report_path = os.path.join(save_dir, 'error_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"错误分析报告已保存: {report_path}")

    return report_path


if __name__ == '__main__':
    vis_dir = os.path.join(MODEL_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    print("="*60)
    print("错误案例分析")
    print("="*60)

    # 加载模型
    device = torch.device('cpu')
    print("\n加载ResNetTransfer模型...")
    model = ResNetTransfer(num_classes=len(CLASS_NAMES))
    model_path = os.path.join(MODEL_DIR, 'resnet_transfer_best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载数据
    print("加载测试数据...")
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"测试集: {len(test_loader.dataset)} 样本")

    # 找出错误案例
    print("\n查找错误案例...")
    error_cases = find_error_cases(model, test_loader, device, num_cases=20)
    print(f"找到 {len(error_cases)} 个错误案例")

    # 可视化错误案例
    visualize_error_cases(error_cases, vis_dir)

    # 生成错误分析报告
    generate_error_report(error_cases, vis_dir)

    print("\n分析完成！")
