"""Построение top-k рекомендаций из вероятностей модели."""
import numpy as np
import pandas as pd

from src.data import PRODUCTS

OWNED_COLS = [f"{p}_lag1" for p in PRODUCTS]


def top_k(scores: np.ndarray, owned: np.ndarray, k: int = 7) -> np.ndarray:
    """Индексы k продуктов с наибольшим скором; продукты, которые уже есть, исключаются."""
    scores = np.where(owned == 1, -np.inf, scores)
    order = np.argsort(-scores, axis=1)[:, :k]
    return order


def recommend(model, features: pd.DataFrame, k: int = 7) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(features[model.feature_names_])
    owned = features[OWNED_COLS].fillna(0).to_numpy()
    idx = top_k(proba, owned, k)
    return idx, np.take_along_axis(proba, idx, axis=1)


def popularity_scores(added: pd.DataFrame) -> np.ndarray:
    """Доля добавлений каждого продукта - скор для бейзлайна по популярности."""
    counts = added[PRODUCTS].sum().to_numpy(dtype=float)
    return counts / counts.sum()
