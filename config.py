import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'PlantVillage', 'PlantVillage')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
WEB_DIR = os.path.join(os.path.dirname(__file__), 'web')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)

NUM_CLASSES = 10
CLASS_NAMES = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus'
]

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]