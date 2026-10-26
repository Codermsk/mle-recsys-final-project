"""Сборка признаков и таргета по месяцам.

Задача: для клиента в месяце t предсказать продукты, которые появятся у него
в месяце t+1 и которых не было в месяце t. Признаки считаются по состоянию
на месяц t и предыдущие месяцы (лаги), таргет - по месяцу t+1.
"""
import numpy as np
import pandas as pd

from src.data import PRODUCTS

N_LAGS = 5

PROFILE_COLS = [
    "ncodpers", "fecha_dato", "ind_empleado", "pais_residencia", "sexo", "age",
    "fecha_alta", "ind_nuevo", "antiguedad", "indrel", "indrel_1mes", "tiprel_1mes",
    "indresi", "indext", "canal_entrada", "indfall", "cod_prov",
    "ind_actividad_cliente", "renta", "segmento",
]
CAT_FEATURES = [
    "ind_empleado", "pais_residencia", "sexo", "indrel_1mes", "tiprel_1mes",
    "indresi", "indext", "canal_entrada", "indfall", "segmento", "cod_prov",
]


class MonthlyData:
    """Снимки клиентов, разложенные по месяцам для быстрого доступа к лагам."""

    def __init__(self, df: pd.DataFrame):
        self.months = sorted(df["fecha_dato"].unique())
        self.products = {}
        self.profiles = {}
        for month, part in df.groupby("fecha_dato", observed=True):
            part = part.set_index("ncodpers")
            self.products[month] = part[PRODUCTS]
            self.profiles[month] = part[[c for c in PROFILE_COLS if c != "ncodpers"]]

    def shift(self, month, k: int):
        idx = self.months.index(month) - k
        return self.months[idx] if idx >= 0 else None


def product_lags(data: MonthlyData, month, clients: pd.Index, n_lags: int = N_LAGS) -> pd.DataFrame:
    """Владение продуктами в месяце month и n_lags-1 месяцах до него."""
    blocks = []
    for k in range(n_lags):
        lag_month = data.shift(month, k)
        if lag_month is None:
            block = pd.DataFrame(np.nan, index=clients, columns=PRODUCTS)
        else:
            block = data.products[lag_month].reindex(clients)
        blocks.append(block.add_suffix(f"_lag{k + 1}").astype("float32"))
    lags = pd.concat(blocks, axis=1)

    cur = lags[[f"{p}_lag1" for p in PRODUCTS]].to_numpy()
    lags["n_products"] = np.nansum(cur, axis=1)
    if n_lags > 1:
        # сколько раз продукты появлялись и пропадали за окно лагов
        stacked = np.stack([lags[[f"{p}_lag{k + 1}" for p in PRODUCTS]].to_numpy()
                            for k in range(n_lags)])
        diff = np.nan_to_num(stacked[:-1]) - np.nan_to_num(stacked[1:])
        lags["n_added_window"] = (diff > 0).sum(axis=(0, 2))
        lags["n_dropped_window"] = (diff < 0).sum(axis=(0, 2))
        for i, p in enumerate(PRODUCTS):
            lags[f"{p}_changes"] = (diff[:, :, i] != 0).sum(axis=0)
    return lags


def build_features(data: MonthlyData, month, clients: pd.Index | None = None,
                   n_lags: int = N_LAGS) -> pd.DataFrame:
    """Признаки клиентов по состоянию на месяц month (рекомендации на следующий месяц)."""
    profile = data.profiles[month]
    if clients is not None:
        profile = profile.reindex(clients)
    clients = profile.index

    feats = pd.DataFrame(index=clients)
    for col in ["age", "antiguedad", "renta", "ind_nuevo", "indrel", "ind_actividad_cliente"]:
        feats[col] = profile[col].astype("float32")
    for col in CAT_FEATURES:
        feats[col] = profile[col].astype(str).replace({"nan": "NA"})

    month_ts = pd.Timestamp(month)
    feats["months_since_alta"] = ((month_ts - profile["fecha_alta"]).dt.days / 30.4).astype("float32")
    feats["alta_month"] = profile["fecha_alta"].dt.month.astype("float32")
    feats["target_month"] = np.float32((month_ts + pd.offsets.MonthBegin(1)).month)

    return feats.join(product_lags(data, month, clients, n_lags))


def added_products(data: MonthlyData, month) -> pd.DataFrame:
    """Матрица новых продуктов в месяце, следующем за month (1 - продукт появился)."""
    next_month = data.shift(month, -1)
    nxt = data.products[next_month]
    cur = data.products[month].reindex(nxt.index)
    common = cur.notna().all(axis=1)
    added = (nxt[common] == 1) & (cur[common] == 0)
    return added.astype("int8")


def build_train_rows(data: MonthlyData, months, n_lags: int = N_LAGS):
    """Обучающая выборка в формате multiclass: одна строка на пару (клиент, новый продукт)."""
    X_parts, y_parts, groups = [], [], []
    for month in months:
        added = added_products(data, month)
        added = added[added.sum(axis=1) > 0]
        pairs = added.stack()
        pairs = pairs[pairs == 1].reset_index()
        pairs.columns = ["ncodpers", "product", "flag"]

        feats = build_features(data, month, pd.Index(pairs["ncodpers"].unique()), n_lags)
        X_parts.append(feats.loc[pairs["ncodpers"]].reset_index(drop=True))
        y_parts.append(pairs["product"].map({p: i for i, p in enumerate(PRODUCTS)}))
        groups.append(pd.Series(month, index=range(len(pairs))))
    return (pd.concat(X_parts, ignore_index=True),
            pd.concat(y_parts, ignore_index=True).astype(int),
            pd.concat(groups, ignore_index=True))
