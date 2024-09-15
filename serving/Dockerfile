FROM python:3.12-slim AS build

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predictor.py .

RUN mkdir models

EXPOSE 8080

CMD ["python3", "predictor.py"]
