FROM python:3.12-slim

WORKDIR /app

COPY registry /app/registry
COPY pyproject.toml /app/pyproject.toml
COPY python /app/python

RUN pip install --no-cache-dir fastapi uvicorn pyjwt python-multipart

ENV REGISTRY_STORAGE=/data
ENV REGISTRY_DEV_MODE=1

CMD ["uvicorn", "registry.app:app", "--host", "0.0.0.0", "--port", "8080"]
