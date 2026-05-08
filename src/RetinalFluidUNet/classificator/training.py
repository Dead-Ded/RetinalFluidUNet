# ==================== Функция загрузки данных ====================
import os

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from .variables import *


def load_data(csv_path, use_original_split=True):
    """
    Загружает CSV с признаками.
    Если use_original_split=True, использует колонку 'subset' для разделения.
    Возвращает X_train, X_val, X_test, y_train, y_val, y_test и список признаков.
    """
    df = pd.read_csv(csv_path)
    print(f"Загружено {len(df)} записей.")
    print(f"Классы и количество:\n{df['disease'].value_counts()}")

    # Исключаем не-признаковые колонки
    meta_cols = ['subset', 'disease', 'patient_id', 'image_num', 'filename']
    feature_cols = [col for col in df.columns if col not in meta_cols]

    X = df[feature_cols]
    y = df['disease']

    # Кодирование меток
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Метки классов: {le.classes_} -> {le.transform(le.classes_)}")

    if use_original_split:
        # Разделение по колонке 'subset'
        train_mask = df['subset'] == 'train'
        val_mask = df['subset'] == 'val'
        test_mask = df['subset'] == 'test'

        X_train = X[train_mask]
        y_train = y_encoded[train_mask]
        X_val = X[val_mask]
        y_val = y_encoded[val_mask]
        X_test = X[test_mask]
        y_test = y_encoded[test_mask]

        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    else:
        # Случайное стратифицированное разбиение
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=TEST_SIZE * 2, stratify=y_encoded, random_state=RANDOM_STATE)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE)
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} (случайное разбиение)")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, le


# ==================== Визуализация EDA ====================
def perform_eda(df, feature_cols, output_dir=None):
    """Строит графики распределения классов, корреляции признаков и boxplots."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. Распределение классов
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='disease', order=df['disease'].value_counts().index)
    plt.title('Распределение классов заболеваний')
    plt.xticks(rotation=45)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 2. Тепловая карта корреляции признаков (выборка 20 наиболее вариативных)
    # Чтобы не загромождать, возьмём топ-20 признаков по дисперсии
    var = df[feature_cols].var().sort_values(ascending=False)
    top_features = var.head(20).index.tolist()
    corr = df[top_features].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Корреляция топ-20 признаков по дисперсии')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 3. Boxplot для нескольких ключевых признаков по классам
    key_feats = [f for f in feature_cols if 'total_area' in f or 'n_cysts' in f]
    if key_feats:
        fig, axes = plt.subplots(1, len(key_feats), figsize=(5 * len(key_feats), 5))
        if len(key_feats) == 1:
            axes = [axes]
        for i, feat in enumerate(key_feats[:5]):  # ограничим 5
            sns.boxplot(data=df, x='disease', y=feat, ax=axes[i])
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'key_features_boxplot.png'), dpi=150, bbox_inches='tight')
        plt.show()


# ==================== Построение моделей ====================
def train_models(X_train, y_train, X_val, y_val, feature_names, output_dir=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'newton-cg', 'sag']  # все поддерживают мультикласс
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'SVC': {
            'model': SVC(random_state=RANDOM_STATE, probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
    }

    # Опционально: XGBoost
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = {
            'model': XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss'),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        }
    except ImportError:
        pass

    best_model = None
    best_score = 0
    best_name = ""
    results = {}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    for name, config in models.items():
        print(f"\n=== Обучение {name} ===")
        grid = GridSearchCV(
            config['model'], config['params'],
            cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1,
            error_score='raise'
        )
        grid.fit(X_train_scaled, y_train)

        val_pred = grid.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        print(f"Лучшие параметры: {grid.best_params_}")
        print(f"Validation Accuracy: {val_acc:.4f}, F1-macro: {val_f1:.4f}")

        results[name] = {
            'best_params': grid.best_params_,
            'best_estimator': grid.best_estimator_,
            'val_accuracy': val_acc,
            'val_f1': val_f1
        }

        if val_f1 > best_score:
            best_score = val_f1
            best_model = grid.best_estimator_
            best_name = name

    print(f"\nЛучшая модель по валидации: {best_name} (F1-macro = {best_score:.4f})")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(best_model, os.path.join(output_dir, f'best_model_{best_name}.pkl'))
        with open(os.path.join(output_dir, 'cv_results.txt'), 'w') as f:
            for name, res in results.items():
                f.write(f"{name}: val_acc={res['val_accuracy']:.4f}, val_f1={res['val_f1']:.4f}\n")
                f.write(f"  Params: {res['best_params']}\n\n")

    return best_model, scaler, best_name, results


# ==================== Оценка на тесте ====================
def evaluate_model(model, scaler, X_test, y_test, le, output_dir=None):
    """Оценка финальной модели на тестовой выборке."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"\nТестовая выборка: Accuracy = {acc:.4f}, F1-macro = {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(8,6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Test Set')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_test.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return acc, f1_macro


# ==================== Важность признаков ====================
def plot_feature_importance(model, feature_names, output_dir=None, top_n=20):
    """Визуализация важности признаков (если модель поддерживает)."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])  # для линейных моделей с одним коэффициентом на класс
    else:
        print("Модель не поддерживает извлечение важности признаков.")
        return

    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Feature Importances')
    plt.barh(range(top_n), importances[indices][::-1], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
    plt.xlabel('Importance')
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()


# ==================== Функция запуска обучения классификатора ====================
# TODO: Переделать под проект
def main(csv: str, output_dir: str, no_original_split: bool = True, skip_eda: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Загрузка данных
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, label_encoder = load_data(
        csv, use_original_split=not no_original_split)

    # 2. EDA (опционально)
    if not skip_eda:
        df_full = pd.read_csv(csv)
        perform_eda(df_full, feature_cols, output_dir=output_dir)

    # 3. Обучение моделей
    best_model, scaler, best_name, cv_results = train_models(
        X_train, y_train, X_val, y_val, feature_cols, output_dir=output_dir
    )

    # 4. Оценка на тесте
    evaluate_model(best_model, scaler, X_test, y_test, label_encoder, output_dir=output_dir)

    # 5. Важность признаков
    plot_feature_importance(best_model, feature_cols, output_dir=output_dir)

    # 6. Сохранение LabelEncoder
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))

    print(f"\nВсе результаты сохранены в {output_dir}")
