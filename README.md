# 番茄叶病识别系统

基于深度学习的番茄叶片病害图像分类项目，支持 **10 类**番茄病害识别，并提供 Web 在线推理与 Grad-CAM 可视化解释。

## 项目简介

本项目使用 [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) 公开数据集中番茄相关子集，训练并对比两类模型：

- **CustomCNN**：自定义卷积神经网络（含残差块 + BatchNorm）
- **ResNetTransfer**：基于 `torchvision.models.resnet50` 的迁移学习（冻结 backbone，替换分类头，分层学习率）

通过 Grad-CAM 热力图给出模型决策依据，方便人工复核。

### 识别的病害类别

```
Tomato_Bacterial_spot                              细菌性斑点病
Tomato_Early_blight                                早疫病
Tomato_healthy                                     健康
Tomato_Late_blight                                 晚疫病
Tomato_Leaf_Mold                                   叶霉病
Tomato_Septoria_leaf_spot                          叶斑病
Tomato_Spider_mites_Two_spotted_spider_mite        螨虫/二斑叶螨
Tomato__Target_Spot                                靶斑病
Tomato__Tomato_YellowLeaf__Curl_Virus              黄化曲叶病毒
Tomato__Tomato_mosaic_virus                        番茄花叶病毒
```

## 技术栈

- **PyTorch** 2.4.1 + **torchvision** 0.19.1
- **ResNet50** 迁移学习 + 自定义 CNN
- **Grad-CAM** 模型可解释性
- **Flask** 3.0.0（后端服务）+ flask-cors
- **HTML / CSS / JavaScript**（前端单页）
- numpy / scikit-learn / Pillow / matplotlib

## 项目结构

```
PlantVillage/
├── app.py                       # Flask Web 服务（推理 + Grad-CAM 可视化）
├── train.py                     # 训练入口（CustomCNN + ResNetTransfer）
├── quick_train.py               # 快速训练（小批量验证流程）
├── config.py                    # 全局配置（路径、类别、超参数、归一化）
├── dataloader.py                # 数据加载 + 增强 + WeightedRandomSampler
├── models.py                    # CustomCNN 与 ResNetTransfer 模型定义
├── error_analysis.py            # 误分析脚本
├── model_visualizer.py          # 模型结构可视化
├── visualize_data.py            # 数据集分布可视化
├── visualize_model_structure.py # 网络结构可视化
├── visualize_training.py        # 训练曲线可视化
├── test_train.py / test_all.py  # 单元测试
├── requirements.txt             # 依赖列表
├── 实验报告.md                    # 实验报告
├── templates/
│   └── index.html               # 前端页面
├── models/                      # 训练好的权重 (.pth)
└── PlantVillage/PlantVillage/   # 数据集（按类别分子目录）
```

## 环境安装

建议 Python 3.8+。

```bash
# 1. 创建虚拟环境（可选）
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
```

> 若需使用 GPU，请根据本机 CUDA 版本到 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应安装命令，覆盖 `requirements.txt` 中的 CPU 版 torch。

## 数据集准备

本项目使用 PlantVillage 数据集的番茄子集。

1. 前往 Kaggle 下载：https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. 解压后，将 10 个番茄类别子目录放入以下结构（注意是两层 `PlantVillage/PlantVillage/`）：

```
d:\PlantVillage\
└── PlantVillage\
    └── PlantVillage\
        ├── Tomato_Bacterial_spot\
        ├── Tomato_Early_blight\
        ├── Tomato_healthy\
        ├── ...
        └── Tomato__Tomato_mosaic_virus\
```

数据加载与划分逻辑见 [dataloader.py](dataloader.py)：默认 **70% 训练 / 15% 验证 / 15% 测试**，训练集使用 `WeightedRandomSampler` 处理类别不平衡。

## 训练模型

```bash
python train.py
```

训练流程：
- 加载数据 → 训练 CustomCNN → 训练 ResNetTransfer → 测试集评估
- 优化器：AdamW + CosineAnnealingLR
- 早停：监控验证准确率，patience=5
- 权重保存至 `models/custom_cnn_best.pth` 与 `models/resnet_transfer_best.pth`

可调超参数见 [config.py](config.py)（IMAGE_SIZE=128、BATCH_SIZE=32、EPOCHS=10、LEARNING_RATE=1e-3）。

> 快速冒烟测试可改用 `python quick_train.py`。

## 运行 Web 服务

```bash
python app.py
```

启动后访问：http://localhost:5000

### 主要 API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/`                | 前端页面 |
| POST | `/api/predict`     | 上传图片 → 返回两类模型的 Top-3 预测 |
| POST | `/api/visualize`   | 上传图片 → 返回 Grad-CAM 热力图（base64 PNG） |
| GET  | `/api/model-info`   | 查询模型参数量、类别列表等元信息 |

请求示例（预测）：

```bash
curl -X POST -F "file=@leaf.jpg" http://localhost:5000/api/predict
```

## 可视化与误分析

```bash
# 数据分布可视化
python visualize_data.py

# 训练曲线（依赖训练日志）
python visualize_training.py

# 模型网络结构可视化
python visualize_model_structure.py

# 误分析（混淆矩阵、错误样本）
python error_analysis.py
```

## 测试

```bash
python test_train.py    # 训练流程单元测试
python test_all.py      # 全量测试
```

## 许可与致谢

- 数据集：[PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)（仅用于学习和研究）
- 预训练权重：基于 ImageNet 的 `torchvision.models.resnet50`
