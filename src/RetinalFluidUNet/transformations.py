from .variables import *

import cv2
import albumentations as A

from torchvision import transforms

# aug_size = 384

# Создаем отдельные трансформации для train, val и test
from torchvision import transforms as T

aug_size = 384

# Базовая детерминированная предобработка
base_preproc = [
    A.MedianBlur(blur_limit=3, p=1.0),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Resize(aug_size, aug_size,
             interpolation=cv2.INTER_LINEAR,
             mask_interpolation=cv2.INTER_NEAREST)
]

# Геометрические аугментации (только train)
train_geom = [
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.75,
             interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
    A.ElasticTransform(alpha=50, sigma=5, p=0.5,
                       interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
]

# Фотометрические аугментации (только train)
train_photo = [
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),
    A.GaussNoise(std_range=(5/255, 15/255), mean_range=(0., 0.), p=0.3),
]

# Полные пайплайны (результат – numpy uint8)
train_transform = A.Compose(
    base_preproc + train_geom + train_photo,
    additional_targets={'mask': 'mask'}
)

val_test_transform = A.Compose(
    base_preproc,
    additional_targets={'mask': 'mask'}
)

inference_image_transform = A.Compose(base_preproc)  # только изображение

tensor_transform = T.Compose([
    T.ToTensor(),                     # uint8 -> float [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

inference_tensor_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])