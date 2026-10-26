"""Метрики качества рекомендаций."""
import numpy as np


def apk(actual: set, predicted: list, k: int = 7) -> float:
    if not actual:
        return 0.0
    score, hits = 0.0, 0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual), k)


def evaluate(actual: list[set], predicted: list[list], n_items: int, k: int = 7) -> dict:
    """Метрики по клиентам, у которых в целевом месяце появился хотя бы один продукт."""
    ap = [apk(a, p, k) for a, p in zip(actual, predicted)]
    hits = [len(a & set(p[:k])) for a, p in zip(actual, predicted)]
    recommended = {item for p in predicted for item in p[:k]}
    return {
        f"map_at_{k}": float(np.mean(ap)),
        f"precision_at_{k}": float(np.sum(hits) / (k * len(predicted))),
        f"recall_at_{k}": float(np.mean([h / len(a) for h, a in zip(hits, actual)])),
        f"hit_rate_at_{k}": float(np.mean([h > 0 for h in hits])),
        "coverage": len(recommended) / n_items,
    }
