from torch.utils.data import DataLoader

from ..datasets import *
from ..transformations import *
from sklearn.model_selection import train_test_split

print("Dataset initialisation started.")

print("FullOCTDataset")
# Инициализация основного датасета без аугментаций (для разделения)
full_dataset = OCTDataset(
    image_dir='.\\DATASET\\ВКРБ\\Данные\\Обучение\\Интра и Суб\\Original',
    mask_dirs={
        'intraretinal': '.\\DATASET\\ВКРБ\\Данные\\Обучение\\Интра и Суб\\Intra',
        'subretinal': '.\\DATASET\\ВКРБ\\Данные\\Обучение\\Интра и Суб\\Sub',
    },
    num_classes=2,
    geom_transform=None,
    photometric_transform=None,
    # n_augs=1
)

# Разделяем на train+val и test (например, 80% train+val, 20% test)
trainval_idx, test_idx = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    random_state=42
)

# Далее разделяем trainval на train и val (например, 80% train, 20% val от trainval)
train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=0.2,
    random_state=42
)

# Создаем датасеты с аугментациями только для обучающей части
train_dataset = OCTSubset(full_dataset, train_idx, geom_photometric_transform=train_transform,
                          tensor_transform=tensor_transform, n_augs=2)
val_dataset = OCTSubset(full_dataset, val_idx, geom_photometric_transform=val_test_transform,
                        tensor_transform=tensor_transform, n_augs=1)
test_dataset = OCTSubset(full_dataset, test_idx, geom_photometric_transform=val_test_transform,
                         tensor_transform=tensor_transform, n_augs=1)

# print(len(train_dataset), len(val_dataset), len(test_dataset))

# DataLoader'ы
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False)

print("RPEDataset")
# Загружаем базовый полный датасет RPE без аугментаций
full_rpe_dataset = OCTDataset(
    image_dir='.\\DATASET\\ВКРБ\\Данные\\Обучение\\ПЭС\\Original',
    mask_dirs={
        'rpe': '.\\DATASET\\ВКРБ\\Данные\\Обучение\\ПЭС\\PES',
    },
    num_classes=1,
    geom_transform=None,
    photometric_transform=None,
    n_augs=1
)

# Разделение на trainval и test для RPE
rpe_trainval_idx, rpe_test_idx = train_test_split(
    range(len(full_rpe_dataset)),
    test_size=0.2,
    random_state=42
)

# Разделение trainval на train и val для RPE
rpe_train_idx, rpe_val_idx = train_test_split(
    rpe_trainval_idx,
    test_size=0.2,
    random_state=42
)

# Повторно используем класс OCTSubset для создания поднаборов с аугментациями и без

rpe_train_dataset = OCTSubset(full_rpe_dataset, rpe_train_idx, geom_photometric_transform=train_transform,
                              tensor_transform=tensor_transform, n_augs=4)

rpe_val_dataset = OCTSubset(full_rpe_dataset, rpe_val_idx, geom_photometric_transform=val_test_transform,
                            tensor_transform=tensor_transform, n_augs=1)

rpe_test_dataset = OCTSubset(full_rpe_dataset, rpe_test_idx, geom_photometric_transform=val_test_transform,
                             tensor_transform=tensor_transform, n_augs=1)

# DataLoaders для RPE
rpe_train_loader = DataLoader(rpe_train_dataset, batch_size=8, shuffle=True, drop_last=True)
rpe_val_loader = DataLoader(rpe_val_dataset, batch_size=8, shuffle=False, drop_last=False)
rpe_test_loader = DataLoader(rpe_test_dataset, batch_size=8, shuffle=False, drop_last=False)
