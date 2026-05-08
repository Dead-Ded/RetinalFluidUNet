import numpy as np
import pandas as pd
from PIL import Image

from .initialisation.device import device
from .transformations import inference_image_transform
from .classificator.models import classifier as classification_model, scaler
from .classificator.variables import *
from .classificator.preprocess import preprocess_image
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

    @torch.no_grad()
    def _predict_single(self, image):
        """Обработка одного изображения (строка, np.ndarray или PIL)."""
        img_np = self._load_image(image)
        input_tensor, img_shape, crop_coords, orig_full_size = preprocess_image(img_np)
        input_tensor = input_tensor.to(self.device)

        seg_probs = self.seg_model(input_tensor)          # (1, C, H, W)
        mask = (seg_probs[0] > 0.5).cpu().numpy().astype(np.uint8)

        features = self.feature_extractor(mask)
        features_scaled = scaler.transform(features)
        pred = self.classifier.predict(features_scaled)

        return pred, mask, features

    @torch.no_grad()
    def _predict_batch(self, images):
        """Обработка списка изображений (каждое — str, np.ndarray или PIL)."""
        input_tensors = []
        for img in images:
            img_np = self._load_image(img)
            tensor, img_shape, crop_coords, orig_full_size = preprocess_image(img_np)
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

        return preds, masks, features_df

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

