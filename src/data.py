"""Загрузка и первичная очистка исходных данных банка."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "train_ver2.csv"
CLEAN_PARQUET = DATA_DIR / "train_clean.parquet"

PRODUCTS = [
    "ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1", "ind_cder_fin_ult1",
    "ind_cno_fin_ult1", "ind_ctju_fin_ult1", "ind_ctma_fin_ult1", "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1", "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1", "ind_plan_fin_ult1",
    "ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1",
    "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1",
]

# Расшифровка продуктов для README и ответов сервиса
PRODUCT_NAMES = {
    "ind_ahor_fin_ult1": "Сберегательный счёт",
    "ind_aval_fin_ult1": "Банковская гарантия",
    "ind_cco_fin_ult1": "Текущий счёт",
    "ind_cder_fin_ult1": "Деривативный счёт",
    "ind_cno_fin_ult1": "Зарплатный счёт",
    "ind_ctju_fin_ult1": "Детский счёт",
    "ind_ctma_fin_ult1": "Особый счёт (más particular)",
    "ind_ctop_fin_ult1": "Счёт particular",
    "ind_ctpp_fin_ult1": "Счёт particular plus",
    "ind_deco_fin_ult1": "Краткосрочный депозит",
    "ind_deme_fin_ult1": "Среднесрочный депозит",
    "ind_dela_fin_ult1": "Долгосрочный депозит",
    "ind_ecue_fin_ult1": "Электронный счёт",
    "ind_fond_fin_ult1": "Фонды",
    "ind_hip_fin_ult1": "Ипотека",
    "ind_plan_fin_ult1": "Пенсионный план",
    "ind_pres_fin_ult1": "Кредит",
    "ind_reca_fin_ult1": "Налоги",
    "ind_tjcr_fin_ult1": "Кредитная карта",
    "ind_valo_fin_ult1": "Ценные бумаги",
    "ind_viv_fin_ult1": "Домашний счёт",
    "ind_nomina_ult1": "Зарплата (payroll)",
    "ind_nom_pens_ult1": "Пенсия (payroll)",
    "ind_recibo_ult1": "Прямой дебет",
}

CAT_COLS = [
    "ind_empleado", "pais_residencia", "sexo", "indrel_1mes", "tiprel_1mes",
    "indresi", "indext", "conyuemp", "canal_entrada", "indfall", "segmento",
]


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    dtypes = {col: "float32" for col in PRODUCTS}
    dtypes.update({
        "ncodpers": "int32", "ind_empleado": "category", "pais_residencia": "category",
        "sexo": "category", "indrel_1mes": "str", "tiprel_1mes": "category",
        "indresi": "category", "indext": "category", "conyuemp": "category",
        "canal_entrada": "category", "indfall": "category", "nomprov": "category",
        "segmento": "category", "age": "str", "antiguedad": "str",
    })
    return pd.read_csv(
        path, dtype=dtypes, parse_dates=["fecha_dato", "fecha_alta"],
        na_values=["NA", " NA", "     NA"], low_memory=False,
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"].str.strip(), errors="coerce").astype("float32")
    df["antiguedad"] = pd.to_numeric(df["antiguedad"].str.strip(), errors="coerce").astype("float32")
    # -999999 в стаже - заглушка, считаем пропуском
    df.loc[df["antiguedad"] < 0, "antiguedad"] = np.nan

    # indrel_1mes хранится вперемешку как 1, 1.0 и '1'
    df["indrel_1mes"] = (
        df["indrel_1mes"].astype(str).str.strip().str.replace(".0", "", regex=False)
        .replace({"nan": np.nan, "P": "5"}).astype("category")
    )
    for col in ["ind_nuevo", "indrel", "tipodom", "cod_prov", "ind_actividad_cliente"]:
        df[col] = df[col].astype("float32")
    df["renta"] = pd.to_numeric(df["renta"], errors="coerce").astype("float32")

    # Пропуски в продуктах (два столбца payroll) - продукта нет
    df[PRODUCTS] = df[PRODUCTS].fillna(0).astype("int8")
    df["ult_fec_cli_1t"] = pd.to_datetime(df["ult_fec_cli_1t"], errors="coerce")
    return df


def load_clean() -> pd.DataFrame:
    return pd.read_parquet(CLEAN_PARQUET)


if __name__ == "__main__":
    raw = load_raw()
    clean(raw).to_parquet(CLEAN_PARQUET, index=False)
    print(f"saved {CLEAN_PARQUET}, rows: {len(raw):,}")
