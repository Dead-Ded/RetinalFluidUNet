import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def visualize_augmentations(image, mask, transform, num_examples=5, figsize=(15, 8)):
    """
    Показывает оригинал и несколько результатов аугментации.

    Параметры:
        image: np.ndarray (H, W, 3) uint8
        mask:  np.ndarray (H, W) или (H, W, 1) uint8
        transform: albumentations Compose (без ToTensor/Normalize,
                   либо с ними – тогда произойдёт автоматическая денормализация)
        num_examples: сколько аугментированных примеров показать
        figsize: размер фигуры
    """
    # Приводим маску к 2D, если она трёхмерная
    if mask.ndim == 3:
        mask = mask.squeeze(-1)

    # Подготовка subplots: верхний ряд – изображения, нижний – маски
    _, axes = plt.subplots(2, num_examples + 1, figsize=figsize)

    # Оригинал
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title("Original Mask")
    axes[1, 0].axis('off')

    for i in range(num_examples):
        # Применяем аугментации
        augmented = transform(image=image, mask=mask)
        img_aug = augmented['image']
        mask_aug = augmented['mask']

        # Если трансформ включает ToTensor и Normalize (на выходе torch.Tensor),
        # денормализуем для визуализации обратно в numpy [0,1]
        if isinstance(img_aug, torch.Tensor):
            # Денормализация (ImageNet статистики)
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img_aug = img_aug.permute(1, 2, 0).numpy()
            img_aug = img_aug * std + mean
            img_aug = np.clip(img_aug, 0, 1)
        else:
            # Если numpy uint8, приводим к float [0,1]
            if img_aug.dtype == np.uint8:
                img_aug = img_aug / 255.0

        if isinstance(mask_aug, torch.Tensor):
            mask_aug = mask_aug.squeeze().numpy()

        # Отрисовка
        axes[0, i+1].imshow(img_aug)
        axes[0, i+1].set_title(f"Aug {i+1}")
        axes[0, i+1].axis('off')

        axes[1, i+1].imshow(mask_aug, cmap='gray')
        axes[1, i+1].set_title(f"Mask {i+1}")
        axes[1, i+1].axis('off')

    plt.tight_layout()
    plt.show()

def final_visualisation(image, prediction, class_names=None):
    img = image.cpu().permute(1, 2, 0).numpy()
    # Отменить нормализацию, если используется стандартная ImageNet нормализация:
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    # Убедиться, что значения в правильном диапазоне
    img = np.clip(img, 0, 1)

    pred = prediction.cpu()

    n_classes = prediction.shape[0]
    class_names_show = class_names if class_names else [f"Class {j + 1}" for j in range(n_classes)]

    plt.figure(figsize=(4 * (n_classes + 1), 4), dpi=100)
    plt.subplot(1, n_classes + 1, 1)
    plt.imshow(img)
    plt.title(f'Input')
    plt.axis('off')

    for j in range(n_classes):
        plt.subplot(1, n_classes + 1, j + 2)
        pred_overlay = pred[j].numpy()

        # Более чёткое отображение
        plt.imshow(img)  # Фон - исходное изображение
        plt.imshow(pred_overlay, alpha=0.5, cmap='Reds')
        plt.title(f'{class_names_show[j]}\nGT (green) & Pred (red)')
        plt.axis('off')

    plt.tight_layout()


def visualisation(image, mask, prediction, class_names=None):
    img = image.cpu().permute(1, 2, 0).numpy()
    # Отменить нормализацию, если используется стандартная ImageNet нормализация:
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    # Убедиться, что значения в правильном диапазоне
    img = np.clip(img, 0, 1)

    mask = mask.cpu()
    pred = prediction.cpu()

    n_classes = mask.shape[0]
    class_names_show = class_names if class_names else [f"Class {j}" for j in range(n_classes)]

    plt.figure(figsize=(4 * (n_classes + 1), 4), dpi=100)
    plt.subplot(1, n_classes + 1, 1)
    plt.imshow(img)
    plt.title(f'Input')
    plt.axis('off')

    for j in range(n_classes):
        plt.subplot(1, n_classes + 1, j + 2)
        mask_overlay = mask[j].numpy()
        pred_overlay = pred[j].numpy()

        # Более четкое отображение
        plt.imshow(img)  # Фон - исходное изображение
        plt.imshow(mask_overlay, alpha=0.5, cmap='Greens')
        plt.imshow(pred_overlay, alpha=0.5, cmap='Reds')
        plt.title(f'{class_names_show[j]}\nGT (green) & Pred (red)')
        plt.axis('off')

    plt.tight_layout()


def visualize_predictions(model: nn.Module, dataloader: DataLoader, device, class_names=None, num_examples=3,
                          save_dir=None):
    """
    model: сегментационная модель
    dataloader: DataLoader
    device: torch.device
    class_names: список названий классов (по умолчанию — индексные)
    num_examples: сколько примеров визуализировать
    save_dir: директория для сохранения (если None, используется plt.show())
    """
    model.eval()
    shown = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            for i in range(images.shape[0]):
                visualisation(images[i], masks[i], preds[i], class_names=class_names)

                # Сохранить или показать
                if save_dir:
                    # Создаем директорию, если она не существует
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f"{save_dir}/pred_{shown}.png", bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()

                shown += 1
                if shown >= num_examples:
                    return


def paint_it_Black(num_epochs: int, train_losses: list[float], val_losses: list[float],
                   train_dices: list[float], val_dices: list[float], train_precision: list[float],
                   val_precision: list[float], train_recall: list[float], val_recall: list[float]):
    # --- ГРАФИКИ ---
    # epochs = range(1, num_epochs+1)  # Это работать не будет, если обучение завершилось досрочно
    epochs = range(1, len(train_losses) + 1)  # Берём длину реально сохранённых метрик

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_dices, label='Train Dice')
    plt.plot(epochs, val_dices, label='Val Dice')
    plt.title("Dice")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_precision, label='Train Precision')
    plt.plot(epochs, val_precision, label='Val Precision')
    plt.title("Precision")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_recall, label='Train Recall')
    plt.plot(epochs, val_recall, label='Val Recall')
    plt.title("Recall")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()
