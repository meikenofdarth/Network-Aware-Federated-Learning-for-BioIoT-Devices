FROM python:3.9-slim
WORKDIR /app

# Added numpy for data processing
RUN pip install grpcio grpcio-tools numpy protobuf

# Copy the Real Data file (Ensure prepare_data.py ran successfully first!)
COPY silo-client/eeg_data.npy . 

COPY silo-client/client.py .
COPY biosignal_pb2.py .
COPY biosignal_pb2_grpc.py .

ENV PYTHONUNBUFFERED=1
CMD ["python", "client.py"]