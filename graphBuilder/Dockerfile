FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["python", "run_pipeline.py"]