FROM python:3.10-slim

WORKDIR /app

# opencv system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# install python deps (cpu-only torch = smaller image)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# copy app code + trained model (dataset NOT included)
COPY app/ app/
COPY templates/ templates/
COPY models/chest_xray_model.pth models/chest_xray_model.pth

RUN mkdir -p data app/uploads

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

EXPOSE 8000

# Render/Railway set PORT env var
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
