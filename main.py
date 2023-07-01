from fastapi import FastAPI, UploadFile, File
from typing import List
import io
import torch
from torchvision.transforms import transforms
from tqdm.notebook import tqdm
from PIL import Image
from ocr import CTCCRNNNoStretchV2, ResizeAndPadHorizontal

from pegon_utils import PEGON_CHARS, CHAR_MAP
from pegon_utils import ctc_collate_fn, CTCDecoder

app = FastAPI()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolo.pt')
ocr_decoder = CTCDecoder.from_path('ocr_state.pt', CTCCRNNNoStretchV2, CHAR_MAP, blank_char=PEGON_CHARS[0])
ocr_decoder.model = ocr_decoder.model.to(DEVICE)

# Define object detection endpoint
@app.post("/infer")
async def detect_objects(file: UploadFile = File(...)):
    # Read image file as bytes
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    # Perform object detection
    result = yolo_model(image).pandas().xyxy[0]
    result = result.sort_values('ymin')

    transform = transforms.Compose([
                          transforms.Grayscale(num_output_channels=1),
                          ResizeAndPadHorizontal(target_h=ocr_decoder.model.image_height, target_w=ocr_decoder.model.image_width),
                          transforms.RandomHorizontalFlip(p=1),
                          transforms.ToTensor(),
                      ])

    line_imgs = []
    for index, row in result.iterrows():
    # Access the column values of each row
        if row['confidence'] > 0.5:
            line_img = image.crop((row['xmin'], row['ymin'], row['xmax'], row['ymax']))
            line_imgs.append(transform(line_img))
    line_imgs = torch.stack(line_imgs).to(DEVICE)

    result = evaluate(ocr_decoder, line_imgs)

    return {'result': result}


def evaluate(decoder, data):
    char_map = decoder.alphabet
    
    return decoder.infer(data.to('cuda'))
