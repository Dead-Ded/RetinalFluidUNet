import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import uniform_filter
from skimage import measure
from tqdm import tqdm

from .classificator.variables import CLASS_NAMES


# ======================== Функция извлечения признаков ========================
def extract_fluid_features(mask, pixel_size_um=1.0):
    """
    Извлечение признаков из бинарной маски жидкостей.

    Parameters
    ----------
    mask : np.array
        Бинарная маска (0/1 или True/False)
    pixel_size_um : float
        Физический размер пикселя в микрометрах

    Returns
    -------
    dict : Словарь с признаками
    """
    features = {}

    # Проверка на пустую маску
    if np.sum(mask) == 0:
        features['total_area_um2'] = 0.0
        features['fluid_fraction'] = 0.0
        features['n_cysts'] = 0
        features['mean_cyst_area'] = 0.0
        features['std_cyst_area'] = 0.0
        features['cv_cyst_area'] = 0.0
        features['max_cyst_area'] = 0.0
        features['median_cyst_circularity'] = 0.0
        features['mean_cyst_circularity'] = 0.0
        features['mean_solidity'] = 0.0
        features['small_cysts_ratio'] = 0.0
        features['centroid_y_norm'] = 0.0
        features['centroid_x_norm'] = 0.0
        features['local_density_std'] = 0.0
        return features

    # Базовые метрики
    features['total_area_um2'] = np.sum(mask) * pixel_size_um ** 2
    features['fluid_fraction'] = np.mean(mask)

    # Компоненты связности
    label_im = measure.label(mask)
    props = measure.regionprops(label_im)

    if props:
        areas = [p.area for p in props]
        perimeters = [p.perimeter for p in props]

        features['n_cysts'] = len(props)
        features['mean_cyst_area'] = np.mean(areas)
        features['std_cyst_area'] = np.std(areas)
        features['cv_cyst_area'] = np.std(areas) / (np.mean(areas) + 1e-7)
        features['max_cyst_area'] = np.max(areas)

        circularities = []
        for area, perimeter in zip(areas, perimeters):
            if perimeter > 0:
                circ = 4 * np.pi * area / (perimeter ** 2)
                circularities.append(circ)
            else:
                circularities.append(0.0)

        features['median_cyst_circularity'] = np.median(circularities)
        features['mean_cyst_circularity'] = np.mean(circularities)

        # Solidity есть в regionprops
        features['mean_solidity'] = np.mean([p.solidity for p in props])

        # Доля мелких кист (площадь < 50 пикселей)
        small_cysts = sum(1 for a in areas if a < 50)
        features['small_cysts_ratio'] = small_cysts / len(areas)

        # Пространственные: средний центроид (нормированный)
        centroids = np.array([p.centroid for p in props])
        features['centroid_y_norm'] = np.mean(centroids[:, 0]) / mask.shape[0]
        features['centroid_x_norm'] = np.mean(centroids[:, 1]) / mask.shape[1]
    else:
        features['n_cysts'] = 0
        features['mean_cyst_area'] = 0.0
        features['std_cyst_area'] = 0.0
        features['cv_cyst_area'] = 0.0
        features['max_cyst_area'] = 0.0
        features['median_cyst_circularity'] = 0.0
        features['mean_cyst_circularity'] = 0.0
        features['mean_solidity'] = 0.0
        features['small_cysts_ratio'] = 0.0
        features['centroid_y_norm'] = 0.0
        features['centroid_x_norm'] = 0.0

    # Текстура: локальная плотность
    local_density = uniform_filter(mask.astype(float), size=10)
    features['local_density_std'] = np.std(local_density[mask > 0]) if np.any(mask) else 0.0

    return features


def extract_multiclass_features(masks: np.ndarray, classes=CLASS_NAMES, pixel_size_um=1.0):
    all_features = {}

    for i, name in enumerate(classes):
        # 1. Извлекаем признаки для конкретного класса
        raw_features = extract_fluid_features(masks[i], pixel_size_um=pixel_size_um)

        # 2. Добавляем префикс класса к каждому ключу и сохраняем в общий словарь
        for key, value in raw_features.items():
            all_features[f"{name}_{key}"] = value

    # 3. Создаем Series (одна колонка/строка) или DataFrame (одна строка)
    # series_result = pd.Series(all_features)
    df_result = pd.DataFrame([all_features])  # Список из одного словаря создает одну строку

    return df_result


# ======================== Основная функция обработки ========================
def process_masks_to_csv(masks_root, output_csv, pixel_size_um=1.0,
                         sample_ratio=1.0, max_files=None,
                         fluid_types=('IRF', 'SRF', 'PED')):
    """
    Обходит папки с масками и сохраняет признаки в CSV.

    Структура папок:
        masks_root/
            train/
                CNV/
                    IRF/
                    SRF/
                    PED/
                DME/ ...
            test/ ...
            val/ ...

    Parameters
    ----------
    masks_root : str
        Корневая папка с масками (например, 'output_masks')
    output_csv : str
        Путь для сохранения CSV файла
    pixel_size_um : float
        Размер пикселя в микрометрах
    sample_ratio : float
        Доля случайных изображений для обработки (0 < ratio <= 1)
    max_files : int or None
        Максимальное количество изображений (если указано, перекрывает sample_ratio)
    fluid_types : tuple
        Список названий папок с жидкостями
    """
    all_rows = []
    subsets = ['train', 'test', 'val']

    for subset in subsets:
        subset_dir = os.path.join(masks_root, subset)
        if not os.path.isdir(subset_dir):
            print(f"Warning: {subset_dir} not found, skipping.")
            continue

        for disease in os.listdir(subset_dir):
            disease_dir = os.path.join(subset_dir, disease)
            if not os.path.isdir(disease_dir):
                continue

            # Собираем список уникальных базовых имён (без расширения)
            # Берём файлы из папки первой жидкости (например, IRF)
            first_fluid = fluid_types[0]
            fluid_dir = os.path.join(disease_dir, first_fluid)
            if not os.path.isdir(fluid_dir):
                print(f"Warning: {fluid_dir} not found, skipping disease {disease}.")
                continue

            all_files = [f for f in os.listdir(fluid_dir) if f.lower().endswith('.png')]
            if not all_files:
                continue

            # Применяем семплирование
            if max_files is not None and len(all_files) > max_files:
                selected_files = random.sample(all_files, max_files)
            elif sample_ratio < 1.0:
                sample_size = max(1, int(len(all_files) * sample_ratio))
                selected_files = random.sample(all_files, sample_size)
            else:
                selected_files = all_files

            print(f"Processing {subset}/{disease} ({len(selected_files)} images)")

            # Обрабатываем каждый файл
            for fname in tqdm(selected_files, desc=f"{subset}/{disease}"):
                base_name = os.path.splitext(fname)[0]
                # Извлекаем ID пациента и номер снимка из имени файла
                # Предполагаемый формат: disease-randomID-imgnum (например, CNV-12345-2)
                # Разделяем по дефисам
                parts = base_name.split('-')
                if len(parts) >= 3:
                    patient_id = parts[1]
                    image_num = parts[2]
                else:
                    patient_id = base_name
                    image_num = '0'

                row = {
                    'subset': subset,
                    'disease': disease,
                    'patient_id': patient_id,
                    'image_num': image_num,
                    'filename': fname
                }

                # Загружаем маски для каждой жидкости
                masks_exist = True
                for fluid in fluid_types:
                    mask_path = os.path.join(disease_dir, fluid, fname)
                    if not os.path.exists(mask_path):
                        print(f"Warning: missing mask {mask_path}, skipping.")
                        masks_exist = False
                        break
                    try:
                        mask_img = Image.open(mask_path).convert('L')
                        mask = np.array(mask_img)
                        # Бинаризуем: > 127
                        mask_bin = (mask > 127).astype(np.uint8)
                        # Извлекаем признаки
                        feats = extract_fluid_features(mask_bin, pixel_size_um=pixel_size_um)
                        # Добавляем в строку с префиксом
                        for key, value in feats.items():
                            row[f'{fluid}_{key}'] = value
                    except Exception as e:
                        print(f"Error processing {mask_path}: {e}")
                        masks_exist = False
                        break

                if masks_exist:
                    all_rows.append(row)

    if not all_rows:
        print("No data collected. Exiting.")
        return

    # Создаём DataFrame и сохраняем
    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
