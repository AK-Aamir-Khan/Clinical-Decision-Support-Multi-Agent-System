# MedAgent-CDSS - educational simulation
# Build:  docker build -t medagent-cdss .
# Run:    docker run --rm -p 8501:8501 --env-file .env medagent-cdss
#         (omit --env-file to run in rule-based mode without an API key)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data ./data

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
