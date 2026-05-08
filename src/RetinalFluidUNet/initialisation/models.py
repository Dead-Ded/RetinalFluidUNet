from sklearn.preprocessing import LabelEncoder

from ..models import RetinalFluidUNet
from ..classificator.variables import CLASS_NAMES
from .device import device

print("Models initialisation started.")

# Инициализация модели для 2 классов
model_main = RetinalFluidUNet(num_classes=2).to(device)

# Инициализация отдельной модели для RPE
model_rpe = RetinalFluidUNet(num_classes=1).to(device)

# le = LabelEncoder()
# cls_enc = le.fit_transform(CLASS_NAMES)
