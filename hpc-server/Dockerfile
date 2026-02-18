FROM python:3.9-slim
WORKDIR /app

# HPC & Azure Requirements
RUN pip install grpcio grpcio-tools onnxruntime numpy \
    azure-digitaltwins-core azure-identity

# Ensure weights directory exists for PVC mount
RUN mkdir -p /app/weights && chmod 777 /app/weights

# Copy all server-side logic
COPY hpc-server/aggregator.py .
COPY hpc-server/fed_avg.py . 
COPY bio_logic.onnx . 
COPY biosignal_pb2.py .
COPY biosignal_pb2_grpc.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 50051
CMD ["python", "aggregator.py"]