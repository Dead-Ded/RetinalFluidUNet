import torch

thresholds = [0.3, 0.4]  # для классов 0 и 1


def dice_coef(preds, targets, thresholds=None, eps=1e-6):
    # Если preds - кортеж, используем только основной выход
    if isinstance(preds, tuple):
        preds = preds[0]

    # Остальной код без изменений
    if thresholds is None:
        thresholds = [0.5]

    if preds.dim() == 3:
        preds = preds.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    preds = torch.sigmoid(preds)
    dices = []
    for i, th in enumerate(thresholds):
        p = (preds[:, i, :, :] > th).float()
        t = targets[:, i, :, :].float()
        intersection = (p * t).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dice = (2. * intersection + eps) / (union + eps)
        dices.append(dice.mean())
    return torch.stack(dices).mean().item()


def accuracy_coef(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    correct = (preds == targets.float()).float()
    return correct.mean().item()


# Аналогично для precision_coef и recall_coef
def precision_coef(preds, targets, threshold=0.5, eps=1e-7):
    if isinstance(preds, tuple):
        preds = preds[0]

    # Остальной код без изменений
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = targets.float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    precision = tp / (tp + fp + eps)
    return precision.item()


def recall_coef(preds, targets, threshold=0.5, eps=1e-7):
    if isinstance(preds, tuple):
        preds = preds[0]

    # Остальной код без изменений
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = targets.float()
    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()
    recall = tp / (tp + fn + eps)
    return recall.item()


def calculate_metrics_per_class(outputs, targets, num_classes):
    """
    Вычисляет метрики для каждого класса, поддерживая deep supervision

    Args:
        outputs: основной выход модели или кортеж (main_output, ds3, ds1)
        targets: целевые маски
        num_classes: количество классов
    """
    metrics = {}

    # Извлекаем основной выход, если передан кортеж
    if isinstance(outputs, tuple):
        main_output = outputs[0]  # Используем только основной выход для метрик
    else:
        main_output = outputs

    # Приводим к правильной размерности
    if main_output.dim() == 3:
        main_output = main_output.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    for class_idx in range(num_classes):
        pred_class = main_output[:, class_idx, :, :]
        target_class = targets[:, class_idx, :, :]
        metrics[f'dice_class_{class_idx}'] = dice_coef(pred_class, target_class)
        metrics[f'precision_class_{class_idx}'] = precision_coef(pred_class, target_class)
        metrics[f'recall_class_{class_idx}'] = recall_coef(pred_class, target_class)
    return metrics


def dice_coeff(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def precision_c(pred, target, smooth=1e-6):
    """
    Precision = TP / (TP + FP)
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    # True Positives: предсказано 1 и правда 1
    tp = (pred * target).sum()
    # False Positives: предсказано 1, а по факту 0
    fp = (pred * (1 - target)).sum()
    return (tp + smooth) / (tp + fp + smooth)


def recall_c(pred, target, smooth=1e-6):
    """
    Recall = TP / (TP + FN)
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    tp = (pred * target).sum()
    if pred.dtype == torch.bool:
        fn = (torch.logical_not(pred) * target).sum()
    else:
        fn = ((1 - pred) * target).sum()
    return (tp + smooth) / (tp + fn + smooth)


def multiclass_dice_coeff(pred, target, smooth=1e-6):
    """
    pred: torch.Tensor of shape (B, C, H, W) - бинарные маски предсказания (0 или 1)
    target: torch.Tensor of shape (B, C, H, W) - бинарные маски таргета (0 или 1)
    Возвращает: средний Dice по всем классам и батчу
    """
    assert pred.shape == target.shape, "Shapes must match!"
    B, C, H, W = pred.shape

    pred = pred.contiguous().view(B, C, -1)
    target = target.contiguous().view(B, C, -1)

    intersection = (pred * target).sum(dim=2)  # (B, C)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2) + target.sum(dim=2) + smooth)  # (B, C)
    return dice.mean()  # средний по батчу и классам
