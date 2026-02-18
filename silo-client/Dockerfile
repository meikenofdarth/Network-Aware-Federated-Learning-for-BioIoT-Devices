FROM python:3.9-slim
WORKDIR /app
RUN pip install grpcio grpcio-tools
# Copy the logic and the generated proto files
COPY silo-client/client.py .
COPY biosignal_pb2.py .
COPY biosignal_pb2_grpc.py .
ENV PYTHONUNBUFFERED=1
CMD ["python", "client.py"]