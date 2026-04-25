FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    python3-tk \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_catboost_cpu_no_svd.json .
COPY heroes_list.json .
COPY teams_list.json .
COPY signs_coder.py .
COPY model_gui.py .

ENV DISPLAY=:0

# 7. Запуск
CMD ["python", "model_gui.py"]