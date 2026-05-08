import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class DiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        targets = targets.float()
        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersection + self.eps) / (union + self.eps)
        loss = 1 - dice
        return loss.mean()


def sobel_edges(tensor, eps=1e-6):
    B, C, H, W = tensor.shape
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=tensor.device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=tensor.device).reshape(1, 1, 3, 3)
    sobel_x = sobel_x.repeat(C, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1)

    # Нормализуем ядра Собеля для сохранения диапазона значений
    sobel_x = sobel_x / 4.0  # Нормализация для сохранения диапазона
    sobel_y = sobel_y / 4.0  # Нормализация для сохранения диапазона

    grad_x = F.conv2d(tensor, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(tensor, sobel_y, padding=1, groups=C)

    # Добавляем стабильности вычислениям
    edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)

    # Нормализуем края к диапазону [0, 1]
    edges_max, _ = edges.view(B, C, -1).max(dim=2, keepdim=True)
    edges_max = edges_max.unsqueeze(3)
    edges = edges / (edges_max + eps)

    return edges


class BoundaryLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        # Для глубокого супервайзинга извлекаем основной выход
        if isinstance(preds, tuple):
            main_output = preds[0]
        else:
            main_output = preds

        preds = torch.sigmoid(main_output)

        # Убеждаемся, что targets находятся в правильном диапазоне
        targets = torch.clamp(targets, 0, 1)

        preds_edges = sobel_edges(preds)
        targets_edges = sobel_edges(targets)

        # Вычисляем Dice-like loss для границ
        intersection = (preds_edges * targets_edges).sum(dim=(2, 3))
        preds_sum = preds_edges.sum(dim=(2, 3))
        targets_sum = targets_edges.sum(dim=(2, 3))

        # Модифицированная формула для избежания отрицательных значений
        dice_coeff = (2. * intersection + self.eps) / (preds_sum + targets_sum + self.eps)
        loss = 1 - dice_coeff

        # Гарантируем, что loss не станет отрицательным
        loss = torch.clamp(loss, min=0, max=1)

        return loss.mean()


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps  # на случай log(0)

    def forward(self, preds, targets):
        """
        preds: logits (не после сигмоид!), shape (B, C, H, W) или (B, 1, H, W)
        targets: binary mask, shape как preds (или будет broadcast)
        """
        # Преобразуем logits к вероятностям
        probs = torch.sigmoid(preds)
        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)  # защита от nan

        targets = targets.float()
        # Для положительного класса
        pt = probs * targets + (1 - probs) * (1 - targets)
        w = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = -w * (1 - pt) ** self.gamma * torch.log(pt)
        return loss.mean()


def deep_supervision_criterion(outputs, masks, focal_loss, dice_loss, weights=(1.0, 0.5, 0.3)):
    """
    Функция потерь для deep supervision

    Args:
        outputs: кортеж (main_output, ds3, ds1)
        masks: целевые маски
        focal_loss: функция focal loss
        dice_loss: функция dice loss
        weights: веса для каждого выхода [main, ds3, ds1]
    """
    if isinstance(outputs, tuple):
        # Deep supervision режим
        main_output, ds3, ds1 = outputs
        loss_main = 0.5 * focal_loss(main_output, masks) + 0.5 * dice_loss(main_output, masks)
        loss_ds3 = 0.5 * focal_loss(ds3, masks) + 0.5 * dice_loss(ds3, masks)
        loss_ds1 = 0.5 * focal_loss(ds1, masks) + 0.5 * dice_loss(ds1, masks)

        return (weights[0] * loss_main + weights[1] * loss_ds3 + weights[2] * loss_ds1) / sum(weights)
    else:
        # Обычный режим (инференс)
        return 0.5 * focal_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)


def compute_class_pos_weights(dataset, batch_size=8, num_workers=0):
    """
    Вычисляет pos_weight для BCEWithLogitsLoss для каждого класса на основе train датасета.
    :param dataset: torch.utils.data.Dataset
    :param batch_size: int
    :param num_workers: int
    :return: torch.Tensor размера (num_classes,)
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    total_pos = None
    total_neg = None
    n_pixels = None

    for _, masks in tqdm(loader, desc='Подсчёт масок'):
        # masks: (B, C, H, W)
        masks = masks.float()
        if total_pos is None:
            C = masks.shape[1]
            total_pos = torch.zeros(C)
            total_neg = torch.zeros(C)
        # Суммируем по всем кроме channel
        pos = masks.view(masks.shape[0], masks.shape[1], -1).sum(dim=(0, 2))
        neg = (1 - masks).view(masks.shape[0], masks.shape[1], -1).sum(dim=(0, 2))
        total_pos += pos.cpu()
        total_neg += neg.cpu()

    # pos_weight = отрицательных / положительных, по PyTorch doc
    pos_weight = total_neg / (total_pos + 1e-6)
    print('Положительных пикселей по классам:', total_pos.numpy())
    print('Отрицательных пикселей по классам:', total_neg.numpy())
    print('Рассчитанные pos_weight:', pos_weight.numpy())
    return pos_weight


def rpe_criterion(outputs, masks):
    if isinstance(outputs, tuple):
        main_output = outputs[0]
    else:
        main_output = outputs

    # Веса можно настроить в зависимости от результатов
    dice_weight = 0.5
    focal_weight = 0.3
    boundary_weight = 0.2

    dice_loss_val = dice_loss(main_output, masks)
    focal_loss_val = focal_loss(main_output, masks)
    boundary_loss_val = boundary_loss(main_output, masks)

    return (dice_weight * dice_loss_val +
            focal_weight * focal_loss_val +
            boundary_weight * boundary_loss_val)


boundary_loss = BoundaryLoss()
focal_loss = FocalLoss(alpha=0.25, gamma=1.5)
dice_loss = DiceLoss()
