FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SIGNSTREAM_BACKEND=torch

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install .
COPY configs ./configs
COPY eval ./eval
RUN mkdir -p artifacts/checkpoints artifacts/onnx && \
    cp eval/fixtures/pose_transformer_signer_holdout.pt artifacts/checkpoints/ && \
    cp eval/fixtures/pose_transformer.onnx artifacts/onnx/pose_transformer.onnx

EXPOSE 8000
CMD ["uvicorn", "signstream.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
