FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU uniquement (evite ~2 Go de wheels NVIDIA dans l'image)
COPY agent/requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements-docker.txt

# Package agent/ pour que "from .preprocessor" fonctionne avec agent.main:app
COPY agent /app/agent
COPY ai_model /app/ai_model

ENV PORT=8000
ENV DRY_RUN=false
ENV K8S_NAMESPACE=default

CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
