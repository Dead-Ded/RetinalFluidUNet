import numpy as np
import pywt
import torch
from scipy.ndimage import uniform_filter
import warnings

from torch import nn

from .initialisation.device import device
from .variables import BASE_DIR


class DnCNN(nn.Module):
    """
    DnCNN с 25 слоями (глубина B = 25).
    Слой 1: Conv2d(in, 64, 3, padding=1) + ReLU
    Слои 2-24: Conv2d(64, 64, 3, padding=1) + BatchNorm2d(64) + ReLU
    Слой 25: Conv2d(64, in, 3, padding=1)   (без BN и ReLU)
    Residual Learning: выход = вход - предсказанный шум.
    """

    def __init__(self, in_channels=1, depth=25, n_channels=64):
        super().__init__()
        layers = []
        # 1-й слой
        layers.append(nn.Conv2d(in_channels, n_channels, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # Скрытые слои (2 .. depth-1)
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        # Выходной слой
        layers.append(nn.Conv2d(n_channels, in_channels, 3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise  # очищенное изображение в sqrt-пространстве


# def anscombe_forward(x, alpha=1.0):
#     """x ∈ [0,1] → y"""
#     return (2.0 / alpha) * np.sqrt(alpha * x + 3.0 / 8.0 * alpha ** 2)
#
#
# def anscombe_inverse(y, alpha=1.0):
#     """y → x (clamp для безопасности)"""
#     x = (alpha / 4.0) * y ** 2 - (3.0 / 8.0) * alpha
#     return np.clip(x, 0, 1)

def anscombe_forward(x, alpha=1.0):
    """x ∈ [0,1] → y"""
    return x


def anscombe_inverse(y, alpha=1.0):
    """y → x (clamp для безопасности)"""
    x = (alpha / 4.0) * y ** 2 - (3.0 / 8.0) * alpha
    return y


def ensure_even(img: np.ndarray) -> np.ndarray:
    """Дополняет изображение отражением до чётных размеров."""
    h, w = img.shape[:2]
    pad_h = 0 if h % 2 == 0 else 1
    pad_w = 0 if w % 2 == 0 else 1
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    return img


def local_ncc_map(f: np.ndarray, g: np.ndarray, win_size=5) -> np.ndarray:
    """
    Локальная карта кросс-корреляции (NCC) между f и g.
    Возвращает массив того же размера со значениями [-1, 1].
    """
    # Преобразуем к float64
    f, g = f.astype(np.float64), g.astype(np.float64)

    # Средние значения в окне
    f_mean = uniform_filter(f, win_size)
    g_mean = uniform_filter(g, win_size)

    # Автокорреляции и кросс-корреляция
    f2_mean = uniform_filter(f ** 2, win_size)
    g2_mean = uniform_filter(g ** 2, win_size)
    fg_mean = uniform_filter(f * g, win_size)

    # Числитель ковариации, знаменатели вариаций
    cov_fg = fg_mean - f_mean * g_mean
    var_f = f2_mean - f_mean ** 2
    var_g = g2_mean - g_mean ** 2

    denom = np.sqrt(np.maximum(var_f * var_g, 0))
    ncc_map = np.divide(cov_fg, denom, out=np.zeros_like(cov_fg), where=denom != 0)
    return ncc_map


wavelet_dncnn = DnCNN(in_channels=1, depth=25, n_channels=64)

# Загружаем сохранённые веса
checkpoint = torch.load(BASE_DIR + r"\data\best_dncnn.pth", map_location=device)
wavelet_dncnn.load_state_dict(checkpoint)
wavelet_dncnn.to(device)
wavelet_dncnn.eval()  # отключаем dropout/batchnorm в режиме обучения


def denoise_wavelet_dncnn(img_gray: np.ndarray) -> np.ndarray:
    img_gray = ensure_even(img_gray)
    h, w = img_gray.shape
    img = img_gray.astype(np.float32) / 255.0
    img = anscombe_forward(img, alpha=1.0)

    wavelet = 'haar'
    # DWT
    LL, (LH, HL, HH) = pywt.dwt2(img, wavelet)
    # SWT (уровень 1)
    swt_result = pywt.swt2(img, wavelet, level=1)[0]  # (LL_swt, (LH_swt, HL_swt, HH_swt))

    def denoise_batch(bands, target_min=2.0 * np.sqrt(3. / 8.), target_max=2.0 * np.sqrt(1. + 3. / 8.)):
        """bands: список 2D numpy массивов одинакового размера"""
        # Нормализация каждого поддиапазона отдельно
        norm_bands = []
        mins, maxs = [], []
        for b in bands:
            b_min, b_max = b.min(), b.max()
            b_norm = (b - b_min) / (b_max - b_min + 1e-8)
            b_scaled = b_norm * (target_max - target_min) + target_min
            norm_bands.append(b_scaled)
            mins.append(b_min)
            maxs.append(b_max)
        # Собираем в батч [B, 1, H, W]
        batch_np = np.stack([b[None, ...] for b in norm_bands], axis=0)  # (B, H, W) -> (B,1,H,W)
        tensor = torch.from_numpy(batch_np).float().to(device)
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            out_tensor = wavelet_dncnn(tensor)  # [B,1,H,W]

        if torch.isnan(out_tensor).any() or torch.isinf(out_tensor).any():
            warnings.warn("Обнаружены NaN/Inf!")

        out_np = out_tensor.cpu().numpy()[:, 0, :, :]  # [B, H, W]

        # Обратная нормализация
        denoised = []
        for i, b in enumerate(out_np):
            out_norm = (b - target_min) / (target_max - target_min)
            out = out_norm * (maxs[i] - mins[i]) + mins[i]
            denoised.append(out.astype(np.float32))

        return denoised

    # DWT поддиапазоны (размер h//2, w//2)
    dwt_bands = [LL, LH, HL, HH]
    LL_d, LH_d, HL_d, HH_d = denoise_batch(dwt_bands)

    # SWT поддиапазоны (размер h, w)
    swt_bands = [swt_result[0], swt_result[1][0], swt_result[1][1], swt_result[1][2]]
    LL_swt_d, LH_swt_d, HL_swt_d, HH_swt_d = denoise_batch(swt_bands)

    # Обратные преобразования
    f_img = pywt.idwt2((LL_d, (LH_d, HL_d, HH_d)), wavelet)
    coeffs_swt_denoised = (LL_swt_d, (LH_swt_d, HL_swt_d, HH_swt_d))
    g_img = pywt.iswt2([coeffs_swt_denoised], wavelet)

    f_img = f_img[:h, :w]
    g_img = g_img[:h, :w]
    f_img = anscombe_inverse(f_img, alpha=1.0)
    g_img = anscombe_inverse(g_img, alpha=1.0)

    # Корреляционное слияние – без изменений
    R_map = local_ncc_map(f_img, g_img, win_size=5)
    R_max = np.max(R_map)
    R0, R1 = 0.2 * R_max, 0.7 * R_max
    mask_high = R_map >= R0
    mask_mid = (R_map >= R1) & ~mask_high
    mask_low = ~(mask_high | mask_mid)
    fused = np.empty_like(f_img)
    fused[mask_high] = 0.5 * f_img[mask_high] + 0.5 * g_img[mask_high]
    fused[mask_mid] = 0.7 * f_img[mask_mid] + 0.3 * g_img[mask_mid]
    fused[mask_low] = 0.3 * f_img[mask_low] + 0.7 * g_img[mask_low]
    fused = np.clip(fused, 0.0, 1.0) * 255.0
    return fused.astype(np.uint8)
