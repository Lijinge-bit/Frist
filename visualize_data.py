"""
数据集可视化脚本
- 展示各类别样本图片
- 绘制类别分布图
- 展示数据增强效果
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image
from torchvision import transforms

from config import DATA_DIR, CLASS_NAMES, MODEL_DIR, IMAGE_SIZE, MEAN, STD

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 类别中文名映射
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


def count_samples():
    class_counts = {}
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = count
    return class_counts


def plot_class_distribution(save_dir):
    """绘制类别分布图"""
    class_counts = count_samples()

    fig, ax = plt.subplots(figsize=(12, 6))

    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    cn_names = [CLASS_NAMES_CN.get(c, c) for c in classes]

    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars = ax.bar(range(len(classes)), counts, color=colors, width=0.7)

    ax.set_title('番茄叶病数据集类别分布', fontsize=14, fontweight='bold')
    ax.set_xlabel('病害类别', fontsize=11)
    ax.set_ylabel('样本数量', fontsize=11)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(cn_names, rotation=45, ha='right', fontsize=9)

    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
               str(count), ha='center', fontsize=9)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"类别分布图已保存: {save_path}")
    return save_path


def plot_sample_images(save_dir):
    """展示各类别样本图片"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()

    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                img_path = os.path.join(class_dir, random.choice(images))
                img = Image.open(img_path).convert('RGB')

                axes[idx].imshow(img)
                cn_name = CLASS_NAMES_CN.get(class_name, class_name)
                axes[idx].set_title(cn_name, fontsize=9, fontweight='bold')
                axes[idx].axis('off')

    plt.suptitle('各类别番茄叶片样本', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'sample_images.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"样本图片已保存: {save_path}")
    return save_path


def plot_data_augmentation(save_dir):
    """展示数据增强效果"""
    # 找一张示例图片
    sample_img = None
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                sample_img = Image.open(os.path.join(class_dir, images[0])).convert('RGB')
                break

    if sample_img is None:
        print("未找到示例图片")
        return

    # 定义各种增强
    augmentations = [
        ('原始图像', transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))])),
        ('随机水平翻转', transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0)
        ])),
        ('随机垂直翻转', transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomVerticalFlip(p=1.0)
        ])),
        ('随机旋转(30°)', transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=30)
        ])),
        ('颜色抖动', transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
        ])),
        ('随机裁剪', transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0))
        ])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (title, transform) in enumerate(augmentations):
        img = transform(sample_img)
        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=10, fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle('数据增强效果展示', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'data_augmentation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"数据增强效果图已保存: {save_path}")
    return save_path


if __name__ == '__main__':
    vis_dir = os.path.join(MODEL_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    print("="*60)
    print("数据集可视化")
    print("="*60)

    print("\n统计类别分布...")
    plot_class_distribution(vis_dir)

    print("\n展示样本图片...")
    plot_sample_images(vis_dir)

    print("\n展示数据增强效果...")
    plot_data_augmentation(vis_dir)

    print("\n所有可视化图表已生成！")
    print(f"保存位置: {vis_dir}")
