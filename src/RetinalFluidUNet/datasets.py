import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def process_mask(mask, num_classes: int):
    mask_np = np.array(mask)
    if num_classes == 1:
        mask_np = (mask_np > 128).astype(np.float32)
    else:
        mask_np = (mask_np > 128).astype(np.float32)  # Или кастомная логика разметки
    return mask_np  # shape: [H, W]


class OCTDataset(Dataset):
    def __init__(self, image_dir, mask_dirs, num_classes,
                 geom_transform=None, photometric_transform=None, n_augs=1):
        self.image_paths = sorted([os.path.join(image_dir, f)
                                   for f in os.listdir(image_dir) if f.endswith('.png')])
        # Тут нет ошибки с путями!
        self.mask_paths = {
            cls: sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')])
            for cls, path in mask_dirs.items()
        }
        self.geom_transform = geom_transform
        self.photometric_transform = photometric_transform
        self.n_augs = n_augs
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths) * self.n_augs

    def __getitem__(self, idx):
        img_idx = idx // self.n_augs
        img = ...
        with Image.open(self.image_paths[img_idx]) as img_file:
            img = np.array(img_file.convert('RGB'))

        # Мультиканальная маска: (C, H, W)
        mask_channels = []
        for cls in self.mask_paths.keys():
            with Image.open(self.mask_paths[cls][img_idx]) as mask_file:
                    mask = mask_file.convert('L')
                    mask_np = process_mask(mask, 1)
                    mask_channels.append(mask_np)
        mask_np = np.stack(mask_channels, axis=-1)  # (H, W, C)

        # Синхронная аугментация
        if self.geom_transform:
            augmented = self.geom_transform(image=img, mask=mask_np)
            img, mask_np = augmented['image'], augmented['mask']
        else:
            pass  # img/mas_np без изменений

        # Фотометрическая аугментация - только к картинке
        if self.photometric_transform:
            img = Image.fromarray(img)
            img = self.photometric_transform(img)  # Tensor [3, H, W]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255  # если нет фотометрии

        # Готовим маску в Tensor: [C, H, W]
        if mask_np.ndim == 2:
            mask_np = mask_np[None, ...]  # (1, H, W)
        else:
            mask_np = mask_np.transpose(2, 0, 1)  # (C, H, W)
        mask_tensor = torch.from_numpy(mask_np.astype(np.float32))

        return img, mask_tensor


# Создаем поднаборы с правильными трансформациями
class OCTSubset(Dataset):
    def __init__(self, base_dataset, indices, geom_photometric_transform=None,
                 tensor_transform=None, n_augs=1):
        self.base_dataset = base_dataset
        self.indices = indices
        self.geom_photometric_transform = geom_photometric_transform  # Albumentations
        self.tensor_transform = tensor_transform          # torchvision (ToTensor + Normalize)
        self.n_augs = n_augs

    def __len__(self):
        return len(self.indices) * self.n_augs

    def __getitem__(self, idx):
        img_idx = idx // self.n_augs
        img, mask = self.base_dataset[self.indices[img_idx]]

        # img: (3,H,W) тензор [0,1] -> numpy (H,W,3) uint8
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img

        # mask: (C,H,W) тензор -> numpy (H,W,C) для Albumentations
        if isinstance(mask, torch.Tensor):
            mask_np = mask.permute(1, 2, 0).numpy()  # (H,W,C)
        else:
            mask_np = mask
        if mask_np.dtype != np.uint8:
            mask_np = mask_np.astype(np.uint8)

        # 1. Albumentations (геометрия + фото) – работает с (H,W,C)
        if self.geom_photometric_transform:
            augmented = self.geom_photometric_transform(image=img_np, mask=mask_np)
            img_np = augmented['image']      # (H,W,3) uint8
            mask_np = augmented['mask']      # (H,W,C) uint8

        # 2. Тензоризация изображения
        if self.tensor_transform:
            img_out = self.tensor_transform(img_np)   # тензор [3,H,W], нормализован
        else:
            img_out = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

        # 3. Маска обратно в (C,H,W) тензор
        if mask_np.ndim == 3:
            mask_out = torch.from_numpy(mask_np.astype(np.float32)).permute(2, 0, 1)
        else:  # на случай, если маска стала одноканальной (H,W)
            mask_out = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)

        return img_out, mask_out


class SingleChannelOCTDataset(Dataset):
    def __init__(self, base_dataset, channel_idx=0,
                 geom_photometric_transform=None, tensor_transform=None, n_augs=1):
        self.base_dataset = base_dataset
        self.channel_idx = channel_idx
        self.geom_photometric_transform = geom_photometric_transform
        self.tensor_transform = tensor_transform
        self.n_augs = n_augs

    def __len__(self):
        return len(self.base_dataset) * self.n_augs

    def __getitem__(self, idx):
        img_idx = idx // self.n_augs
        img, mask_multi = self.base_dataset[img_idx]

        # img -> (H,W,3) uint8
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img

        # Берём нужный канал -> (H,W) uint8
        mask_np = mask_multi[self.channel_idx].numpy().astype(np.uint8)

        # Albumentations
        if self.geom_photometric_transform:
            augmented = self.geom_photometric_transform(image=img_np, mask=mask_np)
            img_np = augmented['image']
            mask_np = augmented['mask']   # остаётся (H,W)

        # Тензоризация изображения
        if self.tensor_transform:
            img_out = self.tensor_transform(img_np)
        else:
            img_out = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

        # Маска -> (1,H,W)
        mask_out = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)

        return img_out, mask_out