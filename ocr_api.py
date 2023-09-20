from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from typing import List
import io
import torch
from torchvision.transforms import transforms
from PIL import Image
from ocr_models import CTCCRNNNoStretchV2 as OCRModel, ResizeAndPadHorizontal
from dotenv import load_dotenv
import os

from pegon_utils import PEGON_CHARS_V2 as PEGON_CHARS, CHAR_MAP_V2 as CHAR_MAP
from pegon_utils import ctc_collate_fn, CTCDecoder

load_dotenv()

if "API_KEY" not in os.environ:
    raise EnvironmentError("API_KEY environment variable is not defined. Please set it before running the application.")
try:
    recog_batch_size = int(os.getenv("RECOG_BATCH_SIZE"))
except:
    raise EnvironmentError(f"RECOG_BATCH_SIZE should be set to int, got {os.getenv('RECOG_BATCH_SIZE')}")

app = FastAPI()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/yolo.pt')
ocr_decoder = CTCDecoder.from_path(model_class=OCRModel,
                                   weight_path='weights/ocr.pt',
                                   model_path='weights/ocr.json',
                                   char_map=CHAR_MAP,
                                   blank_char=PEGON_CHARS[0],
                                   device=DEVICE)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    ResizeAndPadHorizontal(target_h=ocr_decoder.model.image_height, target_w=ocr_decoder.model.image_width),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
])

def get_api_key(api_key: str = Header(default=None)):
    return api_key

# API key dependency
async def check_api_key(api_key: str = Depends(get_api_key)):
    if not api_key or api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get('/')
async def root():
    return {'hello': 'ready for ocr jawi-pegon'}

@app.get("/ping")
async def ping():
    return {"status": "Healthy"}

# Define object detection endpoint
@app.post("/infer")
async def detect_objects(file: UploadFile = File(...), api_key: str = Depends(check_api_key)):
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
        for imgs in torch.split(line_imgs, recog_batch_size):
            result.extend(evaluate(ocr_decoder, imgs))
        return {'result': result}
    except RuntimeError:
        return {'result': [], 'message': "Failed to recognize: Out of memory"}


def evaluate(decoder, data):
    return decoder.infer(data.to('cuda'))
