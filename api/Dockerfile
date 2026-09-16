FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/models/final_fc.keras \
    MODEL_METADATA_PATH=/models/model_metadata.json

WORKDIR /app
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY api/app ./app

EXPOSE 8010
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010", "--proxy-headers"]
