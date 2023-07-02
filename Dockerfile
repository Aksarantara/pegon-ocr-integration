FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \ 
    libglib2.0-0

COPY . .

RUN apt-get install -y git && \
    git clone https://github.com/ultralytics/yolov5.git

RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn main:app --reload

EXPOSE 8000
