FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/hf_cache
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && pip install -r requirements.txt
COPY app ./app
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
