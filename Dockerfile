FROM python:3.13.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MIBICI_MODEL_PATH=/app/models/serving \
    OMP_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY mibici_mlops ./mibici_mlops
COPY models/serving ./models/serving
RUN test -f models/serving/MLmodel

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "mibici_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
