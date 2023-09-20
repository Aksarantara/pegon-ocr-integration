FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \ 
    libglib2.0-0

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000
CMD uvicorn ocr_api:app --reload --host 0.0.0.0 --port 8000

EXPOSE 8000
