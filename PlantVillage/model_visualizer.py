"""
模型可视化模块 - 生成网络结构图和Grad-CAM热力图
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from config import MODEL_DIR, CLASS_NAMES, IMAGE_SIZE


class ModelStructureVisualizer:
    """模型结构可视化器"""
    
    def __init__(self, model, model_name="CustomCNN"):
        self.model = model
        self.model_name = model_name
        self.layer_info = []
        
    def analyze_model(self):
        """分析模型结构"""
        print(f"\n{'='*60}")
        print(f"模型结构分析: {self.model_name}")
        print(f"{'='*60}")
        
        total_params = 0
        trainable_params = 0
        
        for name, module in self.model.named_modules():
            layer_type = type(module).__name__
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if params > 0:
                self.layer_info.append({
                    'name': name,
                    'type': layer_type,
                    'params': params,
                    'trainable': trainable
                })
                total_params += params
                trainable_params += trainable
                
                print(f"{layer_type:25s} | {name:45s} | 参数: {params:>10,}")
        
        print(f"\n{'='*60}")
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"冻结参数: {total_params - trainable_params:,}")
        print(f"{'='*60}\n")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params
        }
    
    def draw_network_diagram(self, save_path=None):
        """绘制网络结构图"""
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 60)
        ax.axis('off')
        ax.set_title(f'{self.model_name} 网络结构图', fontsize=16, fontweight='bold', pad=20)
        
        # 定义层类型和对应的颜色
        layer_colors = {
            'Conv2d': '#FF6B6B',
            'BatchNorm2d': '#4ECDC4',
            'MaxPool2d': '#45B7D1',
            'AvgPool2d': '#96CEB4',
            'AdaptiveAvgPool2d': '#96CEB4',
            'Dropout': '#FFEAA7',
            'Linear': '#DDA0DD',
            'ResidualBlock': '#98D8C8',
            'Sequential': '#F7DC6F',
            'ReLU': '#BB8FCE'
        }
        
        # 简化的模型层次列表
        layers = self._extract_layers()
        
        y_pos = 50
        x_pos = 5
        layer_height = 6
        layer_width = 12
        layer_spacing = 2
        
        for i, layer in enumerate(layers):
            layer_type = layer['type']
            layer_name = layer['name']
            params = layer['params']
            
            color = layer_colors.get(layer_type, '#CCCCCC')
            
            # 绘制层矩形
            rect = patches.FancyBboxPatch(
                (x_pos, y_pos), layer_width, layer_height,
                boxstyle="round,pad=0.05,rounding_size=0.5",
                facecolor=color, edgecolor='black', linewidth=1.5
            )
            ax.add_patch(rect)
            
            # 添加层名称和参数
            short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
            if len(short_name) > 15:
                short_name = short_name[:12] + '...'
            
            ax.text(x_pos + layer_width/2, y_pos + layer_height/2 + 1,
                   f'{short_name}', ha='center', va='center',
                   fontsize=8, fontweight='bold')
            ax.text(x_pos + layer_width/2, y_pos + layer_height/2 - 1.5,
                   f'{params:,}', ha='center', va='center',
                   fontsize=7, color='gray')
            
            # 绘制连接箭头
            if i > 0:
                ax.annotate('', xy=(x_pos, y_pos + layer_height/2),
                           xytext=(x_pos - layer_spacing, y_pos + layer_height/2),
                           arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
            
            x_pos += layer_width + layer_spacing
            
            # 换行
            if (i + 1) % 7 == 0:
                x_pos = 5
                y_pos -= layer_height + 4
        
        # 添加图例
        legend_elements = [patches.Patch(facecolor=color, edgecolor='black', label=name)
                          for name, color in list(layer_colors.items())[:6] if name in [l['type'] for l in layers]]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"网络结构图已保存: {save_path}")
        
        plt.close()
        
    def _extract_layers(self):
        """提取关键层信息"""
        key_layers = []
        skip_count = 0
        
        for info in self.layer_info:
            # 跳过过于细节的层
            name = info['name']
            if '.res_block' in name or 'res_block' in name:
                if skip_count % 2 == 0:
                    key_layers.append({
                        'type': 'ResidualBlock',
                        'name': name,
                        'params': info['params']
                    })
                skip_count += 1
            elif info['params'] > 0 and info['type'] not in ['ModuleList', 'ModuleDict']:
                key_layers.append(info)
                
                if len(key_layers) > 25:
                    break
        
        return key_layers
    
    def generate_summary_text(self):
        """生成模型摘要文本"""
        summary = []
        summary.append("=" * 70)
        summary.append(f"{self.model_name} 模型结构摘要")
        summary.append("=" * 70)
        
        # 统计各类型层
        layer_counts = {}
        param_counts = {}
        
        for info in self.layer_info:
            layer_type = info['type']
            if layer_type not in layer_counts:
                layer_counts[layer_type] = 0
                param_counts[layer_type] = 0
            layer_counts[layer_type] += 1
            param_counts[layer_type] += info['params']
        
        summary.append("\n层类型统计:")
        summary.append("-" * 70)
        summary.append(f"{'层类型':<25s} {'数量':<10s} {'参数量':<15s}")
        summary.append("-" * 70)
        
        for layer_type, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
            summary.append(f"{layer_type:<25s} {count:<10d} {param_counts[layer_type]:>15,}")
        
        # 模型架构说明
        summary.append("\n模型架构说明:")
        summary.append("-" * 70)
        
        if self.model_name == "CustomCNN":
            summary.append("""
1. 输入层: RGB图像 (3通道)
2. 卷积层: Conv2d(3→32) + BatchNorm2d + ReLU + MaxPool2d
3. 残差块阶段:
   - ResBlock1: 32→64, stride=2 (下采样)
   - ResBlock2: 64→64 (保持分辨率)
   - ResBlock3: 64→128, stride=2
   - ResBlock4: 128→128
   - ResBlock5: 128→256, stride=2
   - ResBlock6: 256→256
4. 全局平均池化: AdaptiveAvgPool2d(1,1)
5. 全连接层: Dropout(0.5) → Linear(256→128) → BatchNorm1d → ReLU
6. 输出层: Linear(128→10)
""")
        elif self.model_name == "ResNetTransfer":
            summary.append("""
1. 骨干网络: ResNet50 (ImageNet预训练权重)
   - 冻结所有卷积层和残差块
   - 仅训练自定义分类头
2. 特征提取: AdaptiveAvgPool2d → 2048维特征向量
3. 分类头:
   - Linear(2048→512) + BatchNorm1d + ReLU + Dropout(0.5)
   - Linear(512→256) + BatchNorm1d + ReLU + Dropout(0.3)
   - Linear(256→10)
""")
        
        summary.append("=" * 70)
        
        return "\n".join(summary)


class GradCAM:
    """Grad-CAM: Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.target_layer = target_layer or self._find_target_layer()
        
    def _find_target_layer(self):
        """自动找到目标层"""
        # 对于CustomCNN，使用最后一个ResidualBlock之后的层
        # 对于ResNet，使用layer4
        for name, module in reversed(list(self.model.named_modules())):
            if 'res_block' in name or 'resblock' in name.lower():
                return name, module
            elif 'layer4' in name:
                return name, module
        return None
        
    def save_gradient(self, grad):
        """保存梯度"""
        self.gradients = grad
        
    def forward(self, x, class_idx=None):
        """前向传播"""
        self.model.eval()
        
        # 启用梯度计算
        x.requires_grad_(True)
        
        # 前向传播
        output = self.model(x)
        
        # 如果没有指定类别，使用预测的类别
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # 计算目标类别的梯度
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        
        return output, class_idx
    
    def generate_cam(self, x, class_idx=None):
        """生成Grad-CAM热力图"""
        # 前向传播，获取目标层的输出和梯度
        feature_maps = None
        gradients = None
        
        def forward_hook(module, input, output):
            nonlocal feature_maps
            feature_maps = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()
        
        # 注册hook
        if self.target_layer:
            layer_name, layer = self.target_layer if isinstance(self.target_layer, tuple) else (None, self.target_layer)
            handle_forward = layer.register_forward_hook(forward_hook)
            handle_backward = layer.register_full_backward_hook(backward_hook)
        
        # 前向传播
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # 反向传播
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)
        
        # 移除hook
        if self.target_layer:
            handle_forward.remove()
            handle_backward.remove()
        
        if feature_maps is None or gradients is None:
            print("警告: 无法获取特征图或梯度，使用备选方案")
            return self._generate_fallback_cam(x)
        
        # 计算权重 (梯度均值)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # 加权求和
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        
        # ReLU激活
        cam = torch.clamp(cam, min=0)
        
        # 归一化
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx
    
    def _generate_fallback_cam(self, x):
        """备选方案：使用梯度作为CAM"""
        x.requires_grad_(True)
        output = self.model(x)
        class_idx = output.argmax(dim=1).item()
        
        # 简单获取最后一层的梯度
        output[0, class_idx].backward()
        
        if x.grad is not None:
            cam = x.grad.abs().mean(dim=1).squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.ones((x.shape[2], x.shape[3]))
        
        return cam, class_idx
    
    def visualize_cam(self, image_path, save_path=None, alpha=0.5):
        """可视化Grad-CAM热力图"""
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        
        # 转换为tensor
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        # 生成CAM
        cam, class_idx = self.generate_cam(input_tensor)
        
        # 调整CAM大小
        cam = Image.fromarray((cam * 255).astype(np.uint8))
        cam = cam.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        cam = np.array(cam) / 255.0
        
        # 创建热力图
        heatmap = plt.cm.jet(cam)[:, :, :3]
        
        # 叠加原图
        result = (original_image / 255.0 * alpha + heatmap * (1 - alpha)).clip(0, 1)
        
        # 绘制结果
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image / 255.0)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM 热力图')
        axes[1].axis('off')
        
        axes[2].imshow(result)
        axes[2].set_title(f'叠加结果\n预测: {CLASS_NAMES[class_idx]}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Grad-CAM可视化已保存: {save_path}")
        
        plt.close()
        
        return result, class_idx


def visualize_model_architecture(model, model_name, save_dir=None):
    """可视化模型架构的便捷函数"""
    visualizer = ModelStructureVisualizer(model, model_name)
    
    # 分析模型
    stats = visualizer.analyze_model()
    
    # 生成摘要
    summary = visualizer.generate_summary_text()
    print(summary)
    
    # 绘制网络结构图
    if save_dir:
        save_path = os.path.join(save_dir, f'{model_name}_structure.png')
        visualizer.draw_network_diagram(save_path)
        
        # 保存文本摘要
        txt_path = os.path.join(save_dir, f'{model_name}_summary.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"模型摘要已保存: {txt_path}")
    
    return visualizer


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from models import CustomCNN, ResNetTransfer
    from torchvision import transforms
    
    # 创建可视化目录
    vis_dir = os.path.join(MODEL_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 可视化CustomCNN
    print("\n" + "="*60)
    print("可视化 CustomCNN 模型")
    print("="*60)
    
    custom_cnn = CustomCNN(num_classes=10)
    custom_cnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'custom_cnn_best.pth'), map_location='cpu'))
    custom_cnn.eval()
    
    visualize_model_architecture(custom_cnn, 'CustomCNN', vis_dir)
    
    # 可视化ResNetTransfer
    print("\n" + "="*60)
    print("可视化 ResNetTransfer 模型")
    print("="*60)
    
    resnet = ResNetTransfer(num_classes=10)
    resnet.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'resnet_transfer_best.pth'), map_location='cpu'))
    resnet.eval()
    
    visualize_model_architecture(resnet, 'ResNetTransfer', vis_dir)
    
    # 测试Grad-CAM
    print("\n" + "="*60)
    print("测试 Grad-CAM 可视化")
    print("="*60)
    
    # 找一张测试图片
    from config import DATA_DIR
    test_images = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
                if len(test_images) >= 1:
                    break
        if len(test_images) >= 1:
            break
    
    if test_images:
        test_img = test_images[0]
        print(f"使用测试图像: {test_img}")
        
        gradcam = GradCAM(custom_cnn)
        result, pred_class = gradcam.visualize_cam(test_img, 
                                                    save_path=os.path.join(vis_dir, 'gradcam_example.png'))
        print(f"预测类别: {CLASS_NAMES[pred_class]}")
    else:
        print("未找到测试图像")
    
    print(f"\n所有可视化结果已保存到: {vis_dir}")
