"""Пайплайн обучения: выборки, обучение CatBoost, оценка и сохранение финальной модели.

Запуск финального обучения: python -m src.train
"""
import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.data import DATA_DIR, PRODUCTS, ROOT, load_clean
from src.features import (CAT_FEATURES, MonthlyData, added_products, build_features,
                          build_train_rows)
from src.metrics import evaluate
from src.recommend import OWNED_COLS, popularity_scores, recommend, top_k

SEED = 42
K = 7
MODEL_DIR = ROOT / "models"
FEATURES_LATEST = DATA_DIR / "features_latest.parquet"

DEFAULT_PARAMS = {
    "loss_function": "MultiClass",
    "iterations": 600,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 3,
    "one_hot_max_size": 255,
    "random_seed": SEED,
    "thread_count": -1,
}


@dataclass
class Split:
    """Месяцы признаков: таргет каждого месяца - подключения в следующем месяце."""
    train: list
    valid: list
    test: list


def make_split(months) -> Split:
    # лагам нужно 4 предыдущих месяца, поэтому первый месяц признаков - май 2015
    return Split(train=months[4:14], valid=[months[14]], test=[months[15]])


def holdout(data: MonthlyData, month):
    """Клиенты, у которых в следующем месяце появился хотя бы один продукт, и их новые продукты."""
    added = added_products(data, month)
    added = added[added.sum(axis=1) > 0]
    features = build_features(data, month, added.index)
    actual = [set(np.flatnonzero(row)) for row in added.to_numpy()]
    return features, actual


def popularity(data: MonthlyData, months) -> np.ndarray:
    return popularity_scores(pd.concat([added_products(data, m) for m in months]))


def evaluate_popularity(pop: np.ndarray, features: pd.DataFrame, actual) -> dict:
    owned = features[OWNED_COLS].fillna(0).to_numpy()
    predicted = top_k(np.tile(pop, (len(features), 1)), owned, K)
    return evaluate(actual, predicted.tolist(), len(PRODUCTS), K)


def fit(X, y, X_valid=None, y_valid=None, params=None, verbose=100) -> CatBoostClassifier:
    params = {**DEFAULT_PARAMS, **(params or {})}
    model = CatBoostClassifier(**params, class_names=list(range(len(PRODUCTS))),
                               cat_features=[c for c in CAT_FEATURES if c in X.columns],
                               verbose=verbose, allow_writing_files=False)
    eval_set = (X_valid, y_valid) if X_valid is not None else None
    model.fit(X, y, eval_set=eval_set, early_stopping_rounds=50 if eval_set else None)
    return model


def evaluate_model(model, features: pd.DataFrame, actual) -> dict:
    predicted, _ = recommend(model, features, K)
    return evaluate(actual, predicted.tolist(), len(PRODUCTS), K)


def save_model(model: CatBoostClassifier, pop: np.ndarray, features_month, extra: dict | None = None):
    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(str(MODEL_DIR / "model.bin"))
    meta = {
        "version": datetime.now().strftime("%Y%m%d%H%M"),
        "features_month": str(pd.Timestamp(features_month).date()),
        "feature_names": model.feature_names_,
        "products": PRODUCTS,
        "popularity": pop.round(6).tolist(),
        **(extra or {}),
    }
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def train_final(params: dict, iterations: int, data: MonthlyData | None = None):
    """Обучение на всех доступных месяцах и выгрузка признаков для сервиса.

    Таргет последнего снимка (июнь 2016) неизвестен, поэтому последний месяц признаков
    в обучении - апрель 2016, а признаки мая 2016 идут в хранилище для сервиса.
    """
    data = data or MonthlyData(load_clean())
    months = data.months
    X, y, _ = build_train_rows(data, months[4:-1])
    model = fit(X, y, params={**params, "iterations": iterations})

    latest = months[-1]
    build_features(data, latest).to_parquet(FEATURES_LATEST)
    pop = popularity(data, months[-5:-1])
    meta = save_model(model, pop, latest, {"params": params, "iterations": iterations,
                                           "train_rows": len(X)})
    return model, meta


if __name__ == "__main__":
    best = json.loads((MODEL_DIR / "best_params.json").read_text())
    _, meta = train_final(best["params"], best["iterations"])
    print("saved model", meta["version"], "features month", meta["features_month"])
