import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import NUM_CLASSES

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        residual = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block1 = ResidualBlock(32, 64, stride=2)
        self.res_block2 = ResidualBlock(64, 64)

        self.res_block3 = ResidualBlock(64, 128, stride=2)
        self.res_block4 = ResidualBlock(128, 128)

        self.res_block5 = ResidualBlock(128, 256, stride=2)
        self.res_block6 = ResidualBlock(256, 256)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.res_block1(x)
        x = self.res_block2(x)

        x = self.res_block3(x)
        x = self.res_block4(x)

        x = self.res_block5(x)
        x = self.res_block6(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        # BatchNorm在eval模式下使用全局统计量，单样本推理无问题
        # 训练时跳过batch_size=1的情况避免BN计算错误
        if self.training and x.size(0) == 1:
            pass  # 训练时单样本跳过BN
        else:
            x = self.bn_fc(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

class ResNetTransfer(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, freeze_layers=True):
        super(ResNetTransfer, self).__init__()

        try:
            from torchvision.models import ResNet50_Weights
            self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            self.resnet = models.resnet50(weights=None)

        if freeze_layers:
            # 冻结前面的层，只解冻 layer4 进行微调
            for name, param in self.resnet.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False

        in_features = self.resnet.fc.in_features

        # 自定义分类头
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def get_param_groups(self, lr_pretrained=1e-5, lr_classifier=1e-3):
        """获取分层学习率的参数组，用于优化器"""
        pretrained_params = []
        classifier_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'resnet.fc' in name:
                    classifier_params.append(param)
                else:
                    pretrained_params.append(param)

        return [
            {'params': pretrained_params, 'lr': lr_pretrained, 'name': 'pretrained'},
            {'params': classifier_params, 'lr': lr_classifier, 'name': 'classifier'}
        ]

    def forward(self, x):
        return self.resnet(x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    custom_cnn = CustomCNN().to(device)
    resnet_transfer = ResNetTransfer().to(device)

    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    custom_output = custom_cnn(dummy_input)
    resnet_output = resnet_transfer(dummy_input)

    print(f"CustomCNN output shape: {custom_output.shape}")
    print(f"ResNetTransfer output shape: {resnet_output.shape}")

    total_params_cnn = sum(p.numel() for p in custom_cnn.parameters())
    total_params_resnet = sum(p.numel() for p in resnet_transfer.parameters())

    print(f"CustomCNN total parameters: {total_params_cnn:,}")
    print(f"ResNetTransfer total parameters: {total_params_resnet:,}")