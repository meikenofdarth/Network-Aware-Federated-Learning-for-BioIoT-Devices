FROM python:3.9-slim
WORKDIR /app

# HPC & Azure Requirements
RUN pip install grpcio grpcio-tools onnxruntime numpy \
    azure-digitaltwins-core azure-identity mlflow

# Ensure weights directory exists
RUN mkdir -p /app/weights && chmod 777 /app/weights

COPY hpc-server/aggregator.py .
COPY hpc-server/fed_avg.py . 
# CNE Fix: Wildcard ensures both .onnx and .onnx.data are copied
COPY bio_logic.onnx* . 
COPY biosignal_pb2.py .
COPY biosignal_pb2_grpc.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 50051
CMD ["python", "aggregator.py"]