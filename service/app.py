"""Веб-сервис рекомендаций банковских продуктов."""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, Query
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.data import PRODUCT_NAMES, PRODUCTS
from src.recommend import OWNED_COLS, top_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("recsys")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
FEATURES_PATH = Path(os.getenv("FEATURES_PATH", "data/features_latest.parquet"))
DEFAULT_K = 7
MAX_BATCH = 1000

# --- метрики Prometheus -------------------------------------------------------
REQUESTS = Counter("recsys_recommendation_requests_total",
                   "Запросы рекомендаций по источнику ответа", ["source"])
RECOMMENDED = Counter("recsys_recommended_products_total",
                      "Сколько раз продукт попал в выдачу", ["product"])
TOP1_SCORE = Histogram("recsys_top1_score", "Скор первого продукта в выдаче",
                       buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
INFERENCE_TIME = Histogram("recsys_inference_seconds", "Время инференса модели",
                           buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
BATCH_SIZE = Histogram("recsys_batch_size", "Число клиентов в одном запросе",
                       buckets=[1, 2, 5, 10, 50, 100, 500, 1000])
OWNED_PRODUCTS = Histogram("recsys_client_owned_products", "Число продуктов у клиента в запросе",
                           buckets=[0, 1, 2, 3, 4, 5, 7, 10, 15])
MODEL_INFO = Gauge("recsys_model_info", "Загруженная модель", ["version", "features_month"])
KNOWN_CLIENTS = Gauge("recsys_known_clients", "Клиентов в хранилище признаков")


class Recommender:
    def __init__(self, model_dir: Path, features_path: Path):
        self.model = CatBoostClassifier()
        self.model.load_model(str(model_dir / "model.bin"))
        self.meta = json.loads((model_dir / "model_meta.json").read_text())
        self.popularity = np.array(self.meta["popularity"], dtype=float)
        self.features = pd.read_parquet(features_path)[self.meta["feature_names"]]
        logger.info("model %s loaded, %d clients in feature store",
                    self.meta["version"], len(self.features))

    def recommend(self, client_ids: list[int], k: int) -> dict[int, list[dict]]:
        known = [c for c in client_ids if c in self.features.index]
        result = {}
        if known:
            feats = self.features.loc[known]
            start = time.perf_counter()
            proba = self.model.predict_proba(feats)
            INFERENCE_TIME.observe(time.perf_counter() - start)
            owned = feats[OWNED_COLS].fillna(0).to_numpy()
            idx = top_k(proba, owned, k)
            for row, client in enumerate(known):
                result[client] = self._format(idx[row], proba[row])
                OWNED_PRODUCTS.observe(owned[row].sum())
                TOP1_SCORE.observe(proba[row, idx[row, 0]])
            REQUESTS.labels(source="model").inc(len(known))

        # новые клиенты без истории получают самые популярные продукты
        unknown = [c for c in client_ids if c not in result]
        if unknown:
            idx = top_k(self.popularity[None, :], np.zeros((1, len(PRODUCTS))), k)[0]
            for client in unknown:
                result[client] = self._format(idx, self.popularity)
            REQUESTS.labels(source="popular").inc(len(unknown))
        return result

    @staticmethod
    def _format(idx, scores) -> list[dict]:
        items = []
        for i in idx:
            product = PRODUCTS[i]
            RECOMMENDED.labels(product=product).inc()
            items.append({"product": product, "name": PRODUCT_NAMES[product],
                          "score": round(float(scores[i]), 5)})
        return items


recommender: Recommender | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    recommender = Recommender(MODEL_DIR, FEATURES_PATH)
    MODEL_INFO.labels(version=recommender.meta["version"],
                      features_month=recommender.meta["features_month"]).set(1)
    KNOWN_CLIENTS.set(len(recommender.features))
    yield


app = FastAPI(title="Bank products recommender", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


class Product(BaseModel):
    product: str = Field(examples=["ind_recibo_ult1"])
    name: str = Field(examples=["Прямой дебет"])
    score: float = Field(examples=[0.41])


class RecommendationResponse(BaseModel):
    client_id: int
    recommendations: list[Product]


class BatchRequest(BaseModel):
    client_ids: list[int] = Field(min_length=1, max_length=MAX_BATCH, examples=[[1375586, 1050611]])
    k: int = Field(DEFAULT_K, ge=1, le=len(PRODUCTS))


@app.get("/health")
def health():
    return {"status": "ok", "model_version": recommender.meta["version"],
            "features_month": recommender.meta["features_month"]}


@app.get("/recommendations/{client_id}", response_model=RecommendationResponse)
def recommend_one(client_id: int, k: int = Query(DEFAULT_K, ge=1, le=len(PRODUCTS))):
    BATCH_SIZE.observe(1)
    recs = recommender.recommend([client_id], k)
    return {"client_id": client_id, "recommendations": recs[client_id]}


@app.post("/recommendations", response_model=list[RecommendationResponse])
def recommend_batch(request: BatchRequest):
    ids = list(dict.fromkeys(request.client_ids))
    BATCH_SIZE.observe(len(ids))
    recs = recommender.recommend(ids, request.k)
    return [{"client_id": c, "recommendations": recs[c]} for c in ids]
