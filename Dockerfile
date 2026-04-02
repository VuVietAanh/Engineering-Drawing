FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git gcc g++ libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bước 1 — Cài torch CPU trước
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Bước 2 — Cài detectron2 (cần torch có sẵn)
RUN pip install --no-cache-dir \
    'git+https://github.com/facebookresearch/detectron2.git'

# Bước 3 — Cài các thư viện còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]