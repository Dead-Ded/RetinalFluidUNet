import numpy as np
import pywt
import torch
from scipy.ndimage import uniform_filter
import warnings

from torch import nn

from .initialisation.device import device
from .variables import BASE_DIR, WAVELET


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


wavelet_dncnn = DnCNN(in_channels=1, depth=25, n_channels=64)

# Загружаем сохранённые веса
checkpoint = torch.load(BASE_DIR + r"\data\best_dncnn.pth", map_location=device)
wavelet_dncnn.load_state_dict(checkpoint)
wavelet_dncnn.to(device)
wavelet_dncnn.eval()  # отключаем dropout/batchnorm в режиме обучения


class SqrtTransform:
    """Стабилизация дисперсии: y = sqrt(x), x ∈ [0,1]"""

    @staticmethod
    def forward(x):
        return torch.sqrt(x)

    @staticmethod
    def inverse(y):
        return y ** 2


def ensure_even(img: np.ndarray) -> np.ndarray:
    """Дополняет изображение отражением до чётных размеров по высоте и ширине."""
    h, w = img.shape[:2]
    pad_h = 0 if h % 2 == 0 else 1
    pad_w = 0 if w % 2 == 0 else 1
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    return img


def local_ncc_map(f: np.ndarray, g: np.ndarray, win_size=5) -> np.ndarray:
    """Локальная карта коэффициента корреляции Пирсона (NCC)."""
    f = f.astype(np.float64)
    g = g.astype(np.float64)
    f_mean = uniform_filter(f, win_size)
    g_mean = uniform_filter(g, win_size)
    f2_mean = uniform_filter(f ** 2, win_size)
    g2_mean = uniform_filter(g ** 2, win_size)
    fg_mean = uniform_filter(f * g, win_size)
    cov = fg_mean - f_mean * g_mean
    var_f = f2_mean - f_mean ** 2
    var_g = g2_mean - g_mean ** 2
    denom = np.sqrt(np.maximum(var_f * var_g, 0))
    ncc = np.divide(cov, denom, out=np.zeros_like(cov), where=denom != 0)
    return ncc


def denoise_subband(sub_np: np.ndarray) -> np.ndarray:
    """Пропускает один поддиапазон через DnCNN."""
    tensor = torch.from_numpy(sub_np).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    with torch.no_grad():
        denoised = wavelet_dncnn(tensor)
    return denoised.cpu().squeeze().numpy()


def denoise_wavelet_dncnn(img_gray: np.ndarray) -> np.ndarray:
    """
    Полный пайплайн деноисинга OCT-изображения согласно статье.
    img_gray: входное изображение в градациях серого, uint8 (0..255) или float32 (0..1).
    Возвращает изображение uint8 (0..255).
    """
    # Приводим к float [0,1] и обеспечиваем чётные размеры
    if img_gray.dtype == np.uint8:
        img = img_gray.astype(np.float32) / 255.0
    else:
        img = img_gray.astype(np.float32)
    img = ensure_even(img)
    h, w = img.shape

    # VST (sqrt)
    img_vst = np.sqrt(img)

    # ---------- DWT (уровень 1) ----------
    coeffs = pywt.dwt2(img_vst, WAVELET)
    LL, (LH, HL, HH) = coeffs

    # ---------- Денойзинг DWT поддиапазонов ----------
    LL_d = denoise_subband(LL)
    LH_d = denoise_subband(LH)
    HL_d = denoise_subband(HL)
    HH_d = denoise_subband(HH)

    # ---------- SWT на LL ----------
    # Обеспечиваем чётные размеры LL для SWT
    LL_for_swt = LL.copy()
    if LL_for_swt.shape[0] % 2 != 0:
        LL_for_swt = LL_for_swt[:-1, :]
    if LL_for_swt.shape[1] % 2 != 0:
        LL_for_swt = LL_for_swt[:, :-1]
    swt_result = pywt.swt2(LL_for_swt, WAVELET, level=1)[0]  # (LL_s, (LH_s, HL_s, HH_s))
    LL_s, (LH_s, HL_s, HH_s) = swt_result

    # ---------- Денойзинг SWT поддиапазонов ----------
    LL_s_d = denoise_subband(LL_s)
    LH_s_d = denoise_subband(LH_s)
    HL_s_d = denoise_subband(HL_s)
    HH_s_d = denoise_subband(HH_s)

    # ---------- Восстановление пути f (DWT) ----------
    f_vst = pywt.idwt2((LL_d, (LH_d, HL_d, HH_d)), WAVELET)

    # ---------- Восстановление пути g (SWT + DWT детали) ----------
    LL_swt_reconstructed = pywt.iswt2([(LL_s_d, (LH_s_d, HL_s_d, HH_s_d))], WAVELET)
    # Приведение размеров LL_swt_reconstructed к LL_d
    if LL_swt_reconstructed.shape != LL_d.shape:
        min_h = min(LL_swt_reconstructed.shape[0], LL_d.shape[0])
        min_w = min(LL_swt_reconstructed.shape[1], LL_d.shape[1])
        LL_swt_reconstructed = LL_swt_reconstructed[:min_h, :min_w]
        LH_d_adj = LH_d[:min_h, :min_w]
        HL_d_adj = HL_d[:min_h, :min_w]
        HH_d_adj = HH_d[:min_h, :min_w]
    else:
        LH_d_adj, HL_d_adj, HH_d_adj = LH_d, HL_d, HH_d
    g_vst = pywt.idwt2((LL_swt_reconstructed, (LH_d_adj, HL_d_adj, HH_d_adj)), WAVELET)

    # ---------- Обратное VST ----------
    f_img = np.clip(f_vst, 0, None) ** 2
    g_img = np.clip(g_vst, 0, None) ** 2

    # Приведение к общему размеру (из-за возможной нечётности исходного)
    h_out = min(f_img.shape[0], g_img.shape[0], h)
    w_out = min(f_img.shape[1], g_img.shape[1], w)
    f_img = f_img[:h_out, :w_out]
    g_img = g_img[:h_out, :w_out]

    # ---------- Корреляционное слияние ----------
    R_map = local_ncc_map(f_img, g_img, win_size=5)
    R_max = np.max(R_map)
    R0 = 0.7 * R_max  # порог высокого сходства
    R1 = 0.2 * R_max  # порог низкого сходства

    mask_high = R_map >= R0
    mask_mid = (R_map >= R1) & ~mask_high
    mask_low = R_map < R1

    fused = np.empty_like(f_img)
    fused[mask_high] = 0.5 * f_img[mask_high] + 0.5 * g_img[mask_high]
    fused[mask_mid] = 0.7 * f_img[mask_mid] + 0.3 * g_img[mask_mid]
    fused[mask_low] = 0.3 * f_img[mask_low] + 0.7 * g_img[mask_low]

    # Клипинг и перевод в uint8
    fused = np.clip(fused, 0.0, 1.0) * 255.0
    return fused.astype(np.uint8)
