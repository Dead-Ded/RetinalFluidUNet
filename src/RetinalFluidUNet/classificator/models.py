import pickle

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from ..variables import BASE_DIR

print("classificator/models.py")

classifier: GradientBoostingClassifier = joblib.load(BASE_DIR + "\\data\\best_model_GradientBoosting.pkl")

scaler: StandardScaler = joblib.load(BASE_DIR + "\\data\\scaler.pkl")
