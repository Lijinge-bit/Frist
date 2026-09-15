import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

from config import MODEL_DIR, CLASS_NAMES, IMAGE_SIZE, MEAN, STD
from models import CustomCNN, ResNetTransfer

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# 全局模型变量
custom_cnn = None
resnet_transfer = None


def load_models():
    """加载模型，带错误处理"""
    global custom_cnn, resnet_transfer

    # 加载CustomCNN
    cnn_path = os.path.join(MODEL_DIR, 'custom_cnn_best.pth')
    if os.path.exists(cnn_path):
        try:
            custom_cnn = CustomCNN()
            custom_cnn.load_state_dict(torch.load(cnn_path, map_location=device))
            custom_cnn.to(device)
            custom_cnn.eval()
            print(f"✓ CustomCNN loaded from {cnn_path}")
        except Exception as e:
            print(f"✗ Failed to load CustomCNN: {e}")
            custom_cnn = None
    else:
        print(f"✗ CustomCNN model file not found: {cnn_path}")

    # 加载ResNetTransfer
    resnet_path = os.path.join(MODEL_DIR, 'resnet_transfer_best.pth')
    if os.path.exists(resnet_path):
        try:
            resnet_transfer = ResNetTransfer()
            resnet_transfer.load_state_dict(torch.load(resnet_path, map_location=device))
            resnet_transfer.to(device)
            resnet_transfer.eval()
            print(f"✓ ResNetTransfer loaded from {resnet_path}")
        except Exception as e:
            print(f"✗ Failed to load ResNetTransfer: {e}")
            resnet_transfer = None
    else:
        print(f"✗ ResNetTransfer model file not found: {resnet_path}")


# 启动时加载模型
load_models()


class GradCAM:
    """Grad-CAM可视化类"""

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

    def find_target_layer(self):
        """查找目标层，返回module对象"""
        # 优先查找ResidualBlock
        for name, module in reversed(list(self.model.named_modules())):
            if 'res_block' in name or 'resblock' in name.lower():
                return module
        # 对于ResNet，查找layer4
        for name, module in self.model.named_modules():
            if 'layer4' in name:
                return module
        # 默认返回最后一个卷积层
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """生成Grad-CAM热力图"""
        target_layer = self.find_target_layer()

        if target_layer is None:
            return None, None

        handle_forward = target_layer.register_forward_hook(self.forward_hook)
        handle_backward = target_layer.register_full_backward_hook(self.backward_hook)

        # 前向传播
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        handle_forward.remove()
        handle_backward.remove()

        # 计算CAM
        if self.gradients is None or self.activations is None:
            return None, target_class

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class


def predict(image_bytes):
    """预测图片类别"""
    if custom_cnn is None and resnet_transfer is None:
        raise RuntimeError("No models loaded. Please train models first.")

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    result = {}

    with torch.no_grad():
        if custom_cnn is not None:
            cnn_output = custom_cnn(image)
            cnn_probs = torch.softmax(cnn_output, dim=1).cpu().numpy()[0]
            cnn_pred = int(torch.argmax(cnn_output, dim=1).cpu().item())
            cnn_top3 = sorted(enumerate(cnn_probs), key=lambda x: -x[1])[:3]
            result['custom_cnn'] = {
                'prediction': CLASS_NAMES[cnn_pred],
                'confidence': float(cnn_probs[cnn_pred]),
                'top3': [{'class': CLASS_NAMES[i], 'probability': float(p)} for i, p in cnn_top3]
            }

        if resnet_transfer is not None:
            resnet_output = resnet_transfer(image)
            resnet_probs = torch.softmax(resnet_output, dim=1).cpu().numpy()[0]
            resnet_pred = int(torch.argmax(resnet_output, dim=1).cpu().item())
            resnet_top3 = sorted(enumerate(resnet_probs), key=lambda x: -x[1])[:3]
            result['resnet_transfer'] = {
                'prediction': CLASS_NAMES[resnet_pred],
                'confidence': float(resnet_probs[resnet_pred]),
                'top3': [{'class': CLASS_NAMES[i], 'probability': float(p)} for i, p in resnet_top3]
            }

    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        try:
            image_bytes = file.read()
            result = predict(image_bytes)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/api/visualize', methods=['POST'])
def api_visualize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    model_name = request.form.get('model', 'custom_cnn')
    try:
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image_pil).unsqueeze(0).to(device)
        if model_name == 'resnet':
            model = resnet_transfer
        else:
            model = custom_cnn
        if model is None:
            return jsonify({'error': f'Model {model_name} not loaded'}), 400
        # 生成Grad-CAM
        gradcam = GradCAM(model)
        cam, pred_class = gradcam.generate_cam(image_tensor)
        if cam is None:
            return jsonify({'error': 'Failed to generate Grad-CAM'}), 500

        # 调整CAM大小
        cam_resized = Image.fromarray((cam * 255).astype(np.uint8))
        cam_resized = cam_resized.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        cam_array = np.array(cam_resized) / 255.0

        # 生成热力图
        heatmap = plt.cm.jet(cam_array)[:, :, :3]
        original = np.array(image_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))

        # 叠加结果
        overlay = (original / 255.0 * 0.4 + heatmap * 0.6).clip(0, 1)

        # 生成可视化图片
        buf = io.BytesIO()
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(original / 255.0)
        plt.title('原始图像')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title('Grad-CAM 热力图')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title(f'叠加结果\n预测: {CLASS_NAMES[pred_class]}')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()

        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'visualization': f'data:image/png;base64,{img_base64}',
            'prediction': CLASS_NAMES[pred_class],
            'model': model_name
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/model-info')
def api_model_info():
    try:
        info = {'classes': CLASS_NAMES}
        if custom_cnn is not None:
            cnn_params = sum(p.numel() for p in custom_cnn.parameters())
            cnn_trainable = sum(p.numel() for p in custom_cnn.parameters() if p.requires_grad)
            info['custom_cnn'] = {
                'name': 'CustomCNN',
                'total_params': cnn_params,
                'trainable_params': cnn_trainable,
                'description': '自定义CNN模型，带残差连接和BatchNorm'
            }
        else:
            info['custom_cnn'] = {'name': 'CustomCNN', 'error': 'Model not loaded'}
        if resnet_transfer is not None:
            resnet_params = sum(p.numel() for p in resnet_transfer.parameters())
            resnet_trainable = sum(p.numel() for p in resnet_transfer.parameters() if p.requires_grad)
            info['resnet_transfer'] = {
                'name': 'ResNetTransfer',
                'total_params': resnet_params,
                'trainable_params': resnet_trainable,
                'description': 'ResNet50迁移学习模型，冻结特征提取层'
            }
        else:
            info['resnet_transfer'] = {'name': 'ResNetTransfer', 'error': 'Model not loaded'}
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
