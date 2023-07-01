FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN git clone https://github.com/ultralytics/yolov5.git

RUN pip install --no-cache-dir -r requirements.txt