FROM python:3.9-slim
WORKDIR /app

COPY dashboard-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard-api/server.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]