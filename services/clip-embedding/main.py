import logging
import requests
import torch
import clip
from PIL import Image
import logging
from fastapi import FastAPI, File, UploadFile

from dtypes import FeatureModel, TextModel, TextsModel, ImageListModel, ImageModel, ImageUrlModel
from typing import Dict, List, Annotated
from utils import openImageb64
from io import BytesIO

app = FastAPI()
import aiofiles

@app.on_event("startup")
async def on_startup():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device != "cuda": 
        logging.warning("CLIP not running on CUDA!")
        
    model_name = "ViT-L/14@336px"
    model, preprocess = clip.load(model_name, device=device)
    logging.info(f"CLIP {model_name} successfully loaded")
    app.state.device = device
    app.state.model_name = model_name
    app.state.model = model
    app.state.preprocesor = preprocess


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"API Name": app.state.model_name }


@app.post("/api/text/")
def encode_text(data: TextModel) -> FeatureModel:
    text = data.text
    logging.info(f"Encoding text {text}")
    print(f"Encoding text {text}")
    with torch.no_grad():
        text_tokenized = clip.tokenize(text).to(app.state.device)
        text_feature = app.state.model.encode_text(text_tokenized)
        text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)
    # features have type float16
    features = text_feature.cpu().numpy()
    return FeatureModel.from_numpy(features)



@app.post("/api/texts/")
def encode_texts(data: TextsModel) -> FeatureModel:
    texts = data.texts
    texts_tokenized = torch.cat([clip.tokenize(text) for text in texts])
    with torch.no_grad():
        text_feature = app.state.model.encode_text(texts_tokenized.to(app.state.device))
        text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)
    # features have type float16
    features = text_feature.cpu().numpy()
    return FeatureModel.from_numpy(features)

@app.post("/api/image")
async def encode_image(data: Annotated[bytes, File()]) -> FeatureModel:
    image = Image.open(BytesIO(data)).convert('RGB')
    image = app.state.preprocesor(image).unsqueeze(0).to(app.state.device)
    with torch.no_grad():
        image_feature = app.state.model.encode_image(image)
        image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
    # features have type float16
    features = image_feature.cpu().numpy()
    return FeatureModel.from_numpy(features)


@app.post("/api/image/url")
def encode_image_url(data: ImageUrlModel) -> FeatureModel:
    logging.info(f"Encoding image {data.url}")
    image = Image.open(requests.get(data.url, stream=True).raw).convert('RGB')
    image = app.state.preprocesor(image).unsqueeze(0).to(app.state.device)
    with torch.no_grad():
        image_feature = app.state.model.encode_image(image)
        image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
    # features have type float16
    features = image_feature.cpu().numpy()
    return FeatureModel.from_numpy(features)


@app.post("/api/images")
async def encode_images(data: list[bytes]=File()) -> FeatureModel:
    images = []
    for im_data in data:
        if im_data == b'': continue
        images.append(Image.open(BytesIO(im_data)))
    images = torch.stack([
        app.state.preprocesor(image) for image in images
    ]).to(app.state.device)
    with torch.no_grad():
        image_feature = app.state.model.encode_image(images)
        image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
    features = image_feature.detach().cpu()
    return FeatureModel.from_numpy(features)


