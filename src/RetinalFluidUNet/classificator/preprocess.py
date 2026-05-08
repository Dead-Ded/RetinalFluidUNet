import os

import cv2
import torch
from PIL import Image
from tqdm import tqdm

from .variables import *
from ..noise_reduction import *
from ..transformations import inference_image_transform, inference_tensor_transform


def preprocess_image(image: str | np.ndarray, remove_borders_fn=None):
    """
    Возвращает:
        img_tensor: [1, 3, H, W] готовый для модели
        cropped_size: (H, W) после ресайза
        crop_coords: координаты обрезки или None
        orig_full_size: исходный размер до обрезки
    """
    if isinstance(image, str):
        img_bgr = cv2.imread(image)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        img_rgb = image
    else:
        raise TypeError(f"Unsupported type {type(image)}")

    orig_full_size = img_rgb.shape[:2]
    crop_coords = None

    if remove_borders_fn is not None:
        img_rgb, crop_coords = remove_borders_fn(img_rgb, threshold=240, pad=5)

    # Дополнительное шумоподавление (уже есть MedianBlur в пайплайне, но не помешает)
    # img_rgb = cv2.bilateralFilter(img_rgb, 5, 50, 50)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    denoised_gray = denoise_wavelet_dncnn(img_gray)
    img_rgb = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2RGB)

    # Albumentations: MedianBlur, CLAHE, Resize
    augmented = inference_image_transform(image=img_rgb)
    img_np = augmented['image']  # numpy uint8 (INPUT_SIZE, INPUT_SIZE, 3)

    # Тензоризация + нормализация
    img_tensor = inference_tensor_transform(img_np)  # [3, INPUT_SIZE, INPUT_SIZE]

    return img_tensor.unsqueeze(0), img_np.shape[:2], crop_coords, orig_full_size


# ==================== 3. Инференс для одного изображения ====================
def predict_single_image_nn(model, image_tensor, device: torch.device):
    """
    model: CombinedSegmentationModel или любая nn.Module с выходом [B, C, H, W]
    """
    model.eval()
    with torch.inference_mode(), torch.amp.autocast(device.type):
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)  # [1, C, H, W]
        # output уже содержит вероятности после sigmoid/softmax
        # если нужно дополнительно нормализовать (например, сумма вероятностей не равна 1),
        # можно оставить как есть для независимых классов.
        probs = output  # [1, C, H, W]
    return probs


def probs_to_binary_masks(probs, orig_size, threshold=0.5):
    """
    probs: тензор [1, C, H, W] на GPU/CPU
    orig_size: (orig_h, orig_w)
    Возвращает список numpy масок [C, orig_h, orig_w] бинарных (0/1)
    """
    probs = probs.cpu().squeeze(0)  # [C, H, W]
    masks = []
    for c in range(probs.shape[0]):
        # Бинаризация
        mask_c = (probs[c] > threshold).numpy().astype(np.uint8)
        # Возврат к исходному размеру (интерполяция nearest для бинарных масок)
        mask_c = cv2.resize(mask_c, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
        masks.append(mask_c)
    return masks  # list of [orig_h, orig_w]


# ==================== 4. Обход датасета и сохранение масок ====================
import random


def remove_white_borders(image, threshold=240, pad=5):
    """
    Обрезает белые/светлые рамки вокруг изображения.
    image: numpy array (H, W, 3) RGB
    threshold: порог яркости (0-255), выше которого считается фоном
    pad: дополнительный отступ (чтобы не обрезать вплотную к ткани)
    Возвращает:
        cropped_image, crop_coords (y1, y2, x1, x2)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        cropped = image[y1:y2, x1:x2]
        return cropped, (y1, y2, x1, x2)
    else:
        return image, (0, image.shape[0], 0, image.shape[1])


def denoise_image(image, method='wavelet'):
    """
    Подавление шума на ОКТ-изображении.
    image: numpy array (H, W, 3) RGB
    method: 'median', 'bilateral', 'nlm' (non-local means)
    """
    if method == 'median':
        return cv2.medianBlur(image, 3)
    elif method == 'bilateral':
        # Хорошо сохраняет границы
        return cv2.bilateralFilter(image, d=4, sigmaColor=75, sigmaSpace=75)
    elif method == 'nlm':
        # Медленно, но очень качественно
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    elif method == 'wavelet':
        return denoise_wavelet_dncnn(image)
    else:
        return image


# DO NOT FORGET TO ADD MODEL
def process_dataset(sample_ratio=1.01, max_per_class=None, remove_borders=remove_white_borders, combined_model=None):
    model = combined_model
    model.eval()

    for subset in ['train', 'test', 'val']:
        subset_input_dir = os.path.join(ORIGINAL_DATA_DIR, subset)
        if not os.path.isdir(subset_input_dir):
            print(f"Warning: {subset_input_dir} not found, skipping.")
            continue

        disease_dirs = [d for d in os.listdir(subset_input_dir)
                        if os.path.isdir(os.path.join(subset_input_dir, d))]

        for disease in disease_dirs:
            disease_path = os.path.join(subset_input_dir, disease)
            image_files = [f for f in os.listdir(disease_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                continue

            # Семплирование (если нужно)
            if max_per_class is not None and len(image_files) > max_per_class:
                image_files = random.sample(image_files, max_per_class)
            elif sample_ratio < 1.0:
                sample_size = max(1, int(len(image_files) * sample_ratio))
                image_files = random.sample(image_files, sample_size)

            print(f"Processing {subset}/{disease} ({len(image_files)} images)...")

            # Создаём выходные папки для масок
            for cls_name in CLASS_NAMES:
                mask_cls_dir = os.path.join(OUTPUT_MASKS_DIR, subset, disease, cls_name)
                os.makedirs(mask_cls_dir, exist_ok=True)

            # Аккумуляторы батча
            batch_tensors = []  # список тензоров [1, 3, H, W]
            batch_meta = []  # список кортежей (cropped_size, crop_coords, orig_full_size, base_name)

            # Вспомогательная функция сохранения одной предсказанной маски
            def save_masks(probs_single, meta):
                cropped_size, crop_coords, orig_full_size, base_name = meta
                # probs_single: numpy массив [C, H, W] (после sigmoid порог >0.5)
                for cls_idx, cls_name in enumerate(CLASS_NAMES):
                    mask_prob = probs_single[cls_idx]
                    mask_bin = (mask_prob > 0.5).astype(np.uint8)
                    # Ресайз до размера обрезанного изображения
                    mask_cropped = cv2.resize(mask_bin, (cropped_size[1], cropped_size[0]),
                                              interpolation=cv2.INTER_NEAREST)
                    if crop_coords is not None:
                        # Восстановление полного кадра
                        full_mask = np.zeros(orig_full_size, dtype=np.uint8)
                        y1, y2, x1, x2 = crop_coords
                        if mask_cropped.shape != (y2 - y1, x2 - x1):
                            mask_cropped = cv2.resize(mask_cropped, (x2 - x1, y2 - y1),
                                                      interpolation=cv2.INTER_NEAREST)
                        full_mask[y1:y2, x1:x2] = mask_cropped
                        mask_to_save = full_mask
                    else:
                        mask_to_save = mask_cropped

                    mask_img = Image.fromarray(mask_to_save * 255)
                    save_path = os.path.join(OUTPUT_MASKS_DIR, subset, disease, cls_name, f"{base_name}.png")
                    mask_img.save(save_path)

            # Основной цикл по изображениям
            for img_file in tqdm(image_files):
                img_path = os.path.join(disease_path, img_file)
                base_name = os.path.splitext(img_file)[0]

                # Предобработка
                img_tensor, cropped_size, crop_coords, orig_full_size = preprocess_image(
                    img_path, remove_borders_fn=remove_borders
                )
                batch_tensors.append(img_tensor)
                batch_meta.append((cropped_size, crop_coords, orig_full_size, base_name))

                # Если накопили батч – делаем предсказание
                if len(batch_tensors) == BATCH_SIZE:
                    batch = torch.cat(batch_tensors, dim=0).to(DEVICE)  # [B, 3, H, W]
                    with torch.inference_mode(), torch.amp.autocast('cuda'):
                        probs_batch = model(batch)  # [B, C, H, W]
                    probs_np = probs_batch.cpu().numpy()

                    for i, meta in enumerate(batch_meta):
                        save_masks(probs_np[i], meta)

                    # Очистка
                    del batch, probs_batch
                    batch_tensors.clear()
                    batch_meta.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Обработка оставшихся изображений (неполный батч)
            if batch_tensors:
                batch = torch.cat(batch_tensors, dim=0).to(DEVICE)
                with torch.inference_mode():
                    probs_batch = model(batch)
                probs_np = probs_batch.cpu().numpy()
                for i, meta in enumerate(batch_meta):
                    save_masks(probs_np[i], meta)
                del batch, probs_batch
                batch_tensors.clear()
                batch_meta.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("Processing completed!")
