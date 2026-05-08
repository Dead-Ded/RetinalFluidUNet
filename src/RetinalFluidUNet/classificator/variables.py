import torch

from ..variables import aug_size

# ==================== Конфигурация ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = 'path/to/your_segmentation_model.pth'  # путь к обученной модели
ORIGINAL_DATA_DIR = 'DATASET/OCT2017_DATASET'      # папка с train/test/val
OUTPUT_MASKS_DIR = 'DATASET/OCT2017_MASKS_DATASET'           # куда сохранять маски

INPUT_SIZE = aug_size               # размер, использованный при обучении (aug_size)
NUM_CLASSES = 4                     # число классов сегментации (например, 4 жидкости)
CLASS_NAMES = ['IRF', 'SRF', 'PED']  # имена классов для папок масок

RANDOM_STATE = 42
TEST_SIZE = 0.2  # если не использовать оригинальное разбиение

BATCH_SIZE = 8
NUM_WORKERS = 4

# Нормализация (как в оригинальном коде)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
