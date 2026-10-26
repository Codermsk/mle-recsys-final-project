#!/usr/bin/env bash
# Запуск сервера MLflow.
# Метаданные экспериментов хранятся в SQLite, артефакты (модели, графики) -
# в отдельном хранилище, которое сервер раздаёт клиентам через свой API
# (режим --serve-artifacts), поэтому клиентам не нужен прямой доступ к хранилищу.
#
# По умолчанию хранилище артефактов - локальная папка mlflow/artifacts.
# Чтобы хранить артефакты в S3 (Yandex Object Storage, MinIO), задайте переменные:
#   MLFLOW_ARTIFACTS_DESTINATION=s3://<bucket>/mlflow
#   MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
#   AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
#
# Использование: bash scripts/run_mlflow.sh  (UI: http://localhost:5000)
set -euo pipefail

cd "$(dirname "$0")/.."

MLFLOW_DIR="${MLFLOW_DIR:-mlflow}"
MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///${MLFLOW_DIR}/mlflow.db}"
ARTIFACTS_DESTINATION="${MLFLOW_ARTIFACTS_DESTINATION:-${MLFLOW_DIR}/artifacts}"

mkdir -p "${MLFLOW_DIR}/artifacts"

echo "MLflow: backend=${BACKEND_STORE_URI}, artifacts=${ARTIFACTS_DESTINATION}"
exec mlflow server \
    --backend-store-uri "${BACKEND_STORE_URI}" \
    --artifacts-destination "${ARTIFACTS_DESTINATION}" \
    --serve-artifacts \
    --host "${MLFLOW_HOST}" \
    --port "${MLFLOW_PORT}"
