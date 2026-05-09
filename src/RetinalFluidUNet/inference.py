import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .initialisation.device import device
from .transformations import inference_image_transform
from .classificator.models import classifier as classification_model, scaler
from .classificator.variables import *
from .classificator.preprocess import preprocess_image, remove_white_borders
from .feature_extraction import extract_multiclass_features
from .noise_reduction import denoise_wavelet_dncnn


class InferencePipeline:
    def __init__(
        self,
        segmentation_model: torch.nn.Module,
        denoise_fn=denoise_wavelet_dncnn,
        feature_extractor=extract_multiclass_features,
        classifier=classification_model,
        transform=inference_image_transform,
    ):
        self.device = device
        self.seg_model = segmentation_model.to(device).eval()
        self.denoise_fn = denoise_fn
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.transform = transform

    def _load_image(self, image) -> np.ndarray:
        if isinstance(image, str):
            return np.array(Image.open(image).convert('RGB'))
        if isinstance(image, torch.Tensor):
            return image.numpy().astype(np.uint8)
        if isinstance(image, np.ndarray):
            return image

        return image

    def _afterprocess_mask(self, mask_bin, cropped_size, crop_coords, orig_full_size):
        masks_to_return = []
        # probs_single: numpy массив [C, H, W] (после sigmoid порог >0.5)
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            # Ресайз до размера обрезанного изображения
            mask_cropped = cv2.resize(mask_bin[cls_idx], (cropped_size[1], cropped_size[0]),
                                      interpolation=cv2.INTER_NEAREST)
            if crop_coords is not None:
                # Восстановление полного кадра
                full_mask = np.zeros(orig_full_size, dtype=np.uint8)
                y1, y2, x1, x2 = crop_coords
                if mask_cropped.shape != (y2 - y1, x2 - x1):
                    mask_cropped = cv2.resize(mask_cropped, (x2 - x1, y2 - y1),
                                              interpolation=cv2.INTER_NEAREST)
                full_mask[y1:y2, x1:x2] = mask_cropped
                masks_to_return.append(full_mask)
            else:
                masks_to_return.append(mask_cropped)

        return np.stack(masks_to_return, axis=0)

    def _afterprocess_mask_batch(self, mask_bins, cropped_sizes, crop_coords_s, orig_full_sizes):
        if len(mask_bins.shape) < 4:
            if mask_bins.shape == 2:
                mask_bins = np.expand_dims(mask_bins, 0)
            return self._afterprocess_mask(mask_bins, cropped_sizes, crop_coords_s, orig_full_sizes)

        masks_to_return = []
        for i in range(mask_bins.shape[0]):
            mask_bin = mask_bins[i]
            cropped_size = cropped_sizes[i]
            crop_coords = crop_coords_s[i]
            orig_full_size = orig_full_sizes[i]
            mask_to_return = self._afterprocess_mask(mask_bin, cropped_size, crop_coords, orig_full_size)
            masks_to_return.append(mask_to_return)
        return np.stack(masks_to_return, axis=0)

    @torch.no_grad()
    def _predict_single(self, image):
        """Обработка одного изображения (строка, np.ndarray или PIL)."""
        img_np = self._load_image(image)
        input_tensor, cropped_size, crop_coords, orig_full_size = preprocess_image(
            image=img_np,
            remove_borders_fn=remove_white_borders,
        )
        input_tensor = input_tensor.to(self.device)

        seg_probs = self.seg_model(input_tensor)          # (1, C, H, W)
        mask = (seg_probs[0] > 0.5).cpu().numpy().astype(np.uint8)

        features = self.feature_extractor(mask)
        features_scaled = scaler.transform(features)
        pred = self.classifier.predict(features_scaled)

        mask_restored = self._afterprocess_mask(mask, cropped_size, crop_coords, orig_full_size)

        return pred, mask_restored, features

    @torch.no_grad()
    def _predict_batch(self, images):
        """Обработка списка изображений (каждое — str, np.ndarray или PIL)."""
        input_tensors = []
        cropped_size_list = []
        crop_coords_list = []
        orig_full_size_list = []
        for img in images:
            img_np = self._load_image(img)
            tensor, cropped_size, crop_coords, orig_full_size = preprocess_image(
                image=img_np,
                remove_borders_fn=remove_white_borders,
            )
            cropped_size_list.append(cropped_size)
            crop_coords_list.append(crop_coords)
            orig_full_size_list.append(orig_full_size)
            input_tensors.append(tensor)

        batch_tensor = torch.cat(input_tensors, dim=0).to(self.device)
        seg_probs = self.seg_model(batch_tensor)          # (B, C, H, W)
        seg_probs_np = seg_probs.cpu().numpy()

        masks = []
        features_list = []
        for i in range(seg_probs_np.shape[0]):
            mask_i = (seg_probs_np[i] > 0.5).astype(np.uint8)
            masks.append(mask_i)
            features_list.append(self.feature_extractor(mask_i))

        features_df = pd.concat(features_list, ignore_index=True)
        features_scaled = scaler.transform(features_df)
        preds = self.classifier.predict(features_scaled)

        masks = np.stack(masks, axis=0)
        masks_restored = self._afterprocess_mask_batch(masks, cropped_size_list, crop_coords_list, orig_full_size_list)

        return preds, masks_restored, features_df

    def predict(self, images):
        """
        Универсальный инференс для одного или нескольких изображений.

        Параметры
        ----------
        images : str, np.ndarray, PIL.Image или список/кортеж таких элементов
            - Одиночное изображение: путь (str), массив numpy (H,W), (H,W,3) или (H,W,4), PIL.Image.
            - Пакет: список/кортеж путей, массивов или PIL.Image,
                     либо numpy-массив размерности (N, H, W), (N, H, W, 3) или (N, H, W, 4).

        Возвращает
        -------
        Для одиночного:
            pred : np.ndarray (1 элемент)
            mask : np.ndarray (C, H, W) – бинарная маска после порога 0.5
            features : pd.DataFrame (1 строка)
        Для пакета:
            preds : np.ndarray (B,)
            masks : list[np.ndarray] длины B, каждый (C, H, W)
            features_df : pd.DataFrame (B строк)
        """

        # 1. Сразу обрабатываем списки/кортежи как пакет
        if isinstance(images, (list, tuple)):
            return self._predict_batch(images)

        # 2. Одиночный numpy-массив
        if isinstance(images, np.ndarray):
            if images.ndim == 4:
                # Батч формата (N, H, W, C) или (N, C, H, W) – считаем пакетом
                return self._predict_batch(images)
            elif images.ndim in (2, 3):
                return self._predict_single(images)
            else:
                raise ValueError(
                    f"Некорректная размерность numpy-массива: {images.ndim}. "
                    f"Ожидалось 2, 3 или 4."
                )

        # 3. Строка или PIL.Image – одиночное
        if isinstance(images, (str, Image.Image)):
            return self._predict_single(images)

        # 4. Неподдерживаемый тип
        raise TypeError(
            f"Тип {type(images)} не поддерживается. "
            f"Допустимые типы: str, PIL.Image, np.ndarray, list, tuple."
        )

