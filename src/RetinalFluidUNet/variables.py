import time
from pathlib import Path

aug_size = 384
# SAVE_MODEL_PATH = "DEPRECATED"
BASE_DIR = str(Path(__file__).resolve().parent)
# BASE_DIR = r"E:\AI"
SAVE_MODEL_PATH = ".\\SavedModels\\" + time.strftime("%Y_%m_%d-%H_%M_%S") + "\\"
BOUNDARY_WEIGHT = 1.0
FOCAL_WEIGHT = 7.0
DICE_WEIGHT = 2.0
BCE_WEIGHT = 2.5
WEIGHTS_COEFF = 1 / (DICE_WEIGHT + BCE_WEIGHT + BOUNDARY_WEIGHT)
# accumulation_steps = 4
GOOD_MODELS = [
    # r"G:\Users\Neo Daniel\PycharmProjects\NIR\SavedModels\2026_05_03-14_16_19"
    BASE_DIR + r"\data"
]
