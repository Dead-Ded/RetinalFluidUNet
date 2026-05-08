import torch

from .device import device
from ..noise_reduction import DnCNN
from ..variables import BASE_DIR

print("Noise Reduction initialisation started.")

# Создаём экземпляр модели с теми же параметрами, что использовались при обучении
wavelet_dncnn = DnCNN(in_channels=1, depth=25, n_channels=64)
# Если модель обучалась на RGB, измените in_channels=3

# Загружаем сохранённые веса
checkpoint = torch.load(BASE_DIR + r"\data\best_dncnn.pth", map_location=device)
wavelet_dncnn.load_state_dict(checkpoint)
wavelet_dncnn.to(device)
wavelet_dncnn.eval()  # отключаем dropout/batchnorm в режиме обучения
