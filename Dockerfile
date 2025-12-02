# server/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps (if you need gcc/libpq-dev for some packages, keep them)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gcc \
      libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# copy server requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy server source
COPY . .

# create non-root user and chown
RUN useradd --create-home appuser \
 && chown -R appuser:appuser /app

USER appuser

ENV GUNICORN_CMD="gunicorn --bind 0.0.0.0:5174 server.app:app --workers 4 --threads 4 --timeout 120"

EXPOSE 5174

# use the env variable to allow override via docker-compose
CMD sh -c "$GUNICORN_CMD"
