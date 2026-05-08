import torch

from .device import device
from ..models import CombinedSegmentationModel
from ..variables import BASE_DIR


def init_models(model_main_, model_rpe_, path=BASE_DIR + "\\data\\"):
    checkpoint_main = torch.load(path + "IRF_SRF.pth")
    model_main_.load_state_dict(checkpoint_main)
    model_main_.eval()

    checkpoint_rpe = torch.load(path + "PED.pth")
    model_rpe_.load_state_dict(checkpoint_rpe)
    model_rpe_.eval()


def get_combined_model(model_main_, model_rpe_):
    class_configs = {
        'IRF': {
            'model': model_main_,
            'channel': 0,  # берём первый канал из двух
            'activation': 'sigmoid'
        },
        'SRF': {
            'model': model_main_,
            'channel': 1,  # второй канал
            'activation': 'sigmoid'
        },
        'PED': {
            'model': model_rpe_,
            'channel': None,  # одноканальный выход
            'activation': 'sigmoid'
        }
    }
    combined_model = CombinedSegmentationModel(class_configs, device=device.type)
    combined_model.eval()

    return combined_model
