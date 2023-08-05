from fastapi import FastAPI, UploadFile, File
from typing import List
import io
import torch
from torchvision.transforms import transforms
from PIL import Image
from ocr import CTCCRNNNoStretchV2 as OCRModel, ResizeAndPadHorizontal

from pegon_utils import PEGON_CHARS_V2 as PEGON_CHARS, CHAR_MAP_V2 as CHAR_MAP
from pegon_utils import ctc_collate_fn, CTCDecoder

RECOG_BATCH_SIZE = 2

app = FastAPI()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolo.pt')
ocr_decoder = CTCDecoder.from_path('ocr_state.pt', char_map=CHAR_MAP, blank_char=PEGON_CHARS[0])
ocr_decoder.model = ocr_decoder.model.to(DEVICE)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    ResizeAndPadHorizontal(target_h=ocr_decoder.model.image_height, target_w=ocr_decoder.model.image_width),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
])

@app.get("/ping")
async def ping():
    return {"status": "Healthy"}

# Define object detection endpoint
@app.post("/infer")
async def detect_objects(file: UploadFile = File(...)):
    # Read image file as bytes
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    # Perform object detection
    result = []
    try:
        result = yolo_model(image).pandas().xyxy[0]
        result = result.sort_values('ymin')
    except RuntimeError:
        return {'result': [], 'message': "Failed to detect text: Out of memory"}
    if result.empty:
        return {'result': [], 'message': "Detected no text at all"}
    # we shouldn't need to check if the result variable is empty at this point
    line_imgs = []
    for index, row in result.iterrows():
        # Access the column values of each row
        if row['confidence'] > 0.5:
            line_img = image.crop((row['xmin'], row['ymin'], row['xmax'], row['ymax']))
            line_imgs.append(transform(line_img))
    line_imgs = torch.stack(line_imgs).to(DEVICE)
    
    try:
        result = []
        for imgs in torch.split(line_imgs, RECOG_BATCH_SIZE):
            result.extend(evaluate(ocr_decoder, imgs))
        return {'result': result}
    except RuntimeError:
        return {'result': [], 'message': "Failed to recognize: Out of memory"}


def evaluate(decoder, data):
    return decoder.infer(data.to('cuda'))

