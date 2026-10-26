FROM python:3.11-slim

WORKDIR /app

COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

COPY src/ src/
COPY service/ service/
COPY models/ models/

ENV MODEL_DIR=/app/models \
    FEATURES_PATH=/app/data/features_latest.parquet

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
