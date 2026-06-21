FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY training/ ./training/
COPY eval/ ./eval/

RUN pip install --no-cache-dir -e .

ENV SYON_MODELS_DIR=/models
ENV SYON_API_HOST=0.0.0.0
ENV SYON_API_PORT=8000

EXPOSE 8000

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--model", "/models/syon-7b.gguf", "--api-port", "8000"]