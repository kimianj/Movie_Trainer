FROM python:3.12-slim

WORKDIR /app

# libgomp1: faiss-cpu's runtime dependency on slim base images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch wheel first: the unpinned "torch" in requirements.txt would
# otherwise resolve to the default CUDA build (multi-GB) from PyPI. Once the
# CPU wheel satisfies it, the second install is a no-op for torch.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY Data/processed/ Data/processed/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
