FROM python:3.9-slim
WORKDIR /app

# Step 1: Install all PyPI packages first
RUN pip install --no-cache-dir \
    grpcio grpcio-tools onnxruntime numpy \
    azure-digitaltwins-core azure-identity \
    mlflow==2.8.0

# Step 2: Install torch CPU-only separately using extra-index-url
# --extra-index-url adds PyTorch's server alongside PyPI, not instead of it
RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

# Step 3: Install opacus (depends on torch being present first)
RUN pip install --no-cache-dir opacus

RUN mkdir -p /app/weights && chmod 777 /app/weights

COPY hpc-server/aggregator.py .
COPY hpc-server/fed_avg.py .
COPY bio_logic.onnx* .
COPY biosignal_pb2.py .
COPY biosignal_pb2_grpc.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 50051
CMD ["python", "aggregator.py"]