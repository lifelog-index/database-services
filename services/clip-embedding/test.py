import requests
import os
from dotenv import load_dotenv
from pathlib import Path
from io import BytesIO
from PIL import Image
import base64

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)
BATCH_SIZE = 100
CLIP_PORT = os.environ.get("CLIP_PORT", None)

assert CLIP_PORT is not None, "CLIP_PORT is not set"

def test_encode_single_text(text = "a funny man with a hat"):
    api = f"api/text"
    url = f"http://localhost:{CLIP_PORT}/{api}"
    response = requests.post(url, json={"text": text})
    # assert response.status_code == 200
    response = response.json() 
    print(response)
    assert response['feature'] is not None

def test_encode_batch_text(text = "a funny man with a hat 1"):
    api = f"api/texts"
    url = f"http://localhost:{CLIP_PORT}/{api}"
    response = requests.post(url, json={"texts": [text]*BATCH_SIZE})
    # assert response.status_code == 200
    response = response.json() 
    print(response)
    assert response['feature'] is not None

def test_encode_image_url(image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png"):
    api = f"api/image/url"
    url = f"http://localhost:{CLIP_PORT}/{api}"
    response = requests.post(url, json={"url": image_url})
    response = response.json() 
    print(response)
    # assert response.status_code == 201, f"Request on {url} failed"
    assert response['feature'] is not None
    return response['feature']

def PIL2b64(img):
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  
    im_b64 = base64.b64encode(im_bytes)
    return im_b64.decode()

def test_encode_single_image_old(image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png"):
    api = f"api/image/"
    url = f"http://localhost:{CLIP_PORT}/{api}"
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    imageb64 = PIL2b64(img=image)
    response = requests.post(url, json={"imageb64": imageb64})
    response = response.json() 
    print(response)
    # assert response.status_code == 201, f"Request on {url} failed"
    assert response['feature'] is not None
    return response['feature']


def test_encode_batch_image_old(image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png"):
    api = f"api/images"
    url = f"http://localhost:{CLIP_PORT}/{api}"
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    imageb64 = PIL2b64(img=image)
    imageb64_ls = [imageb64] * BATCH_SIZE
    response = requests.post(url, json={"imageb64_ls": imageb64_ls})
    response = response.json() 
    # print(response)
    # assert response.status_code == 201, f"Request on {url} failed"
    assert response['feature'] is not None
    return response['feature']


def test_encode_single_image(image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png"):
    api = f"api/image/"
    url = f"http://localhost:{CLIP_PORT}/{api}"
    image_bytes = requests.get(image_url, stream=True).raw
    response = requests.post(url=url, files=[('data', image_bytes)])
    response = response.json() 
    print(response)
    # assert response.status_code == 201, f"Request on {url} failed"
    assert response['feature'] is not None
    return response['feature']


def test_encode_batch_image(image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png"):
    api = f"api/images"
    import copy
    url = f"http://localhost:{CLIP_PORT}/{api}"
    files = [] 
    for i in range(2):
        image_bytes = requests.get(image_url, stream=True).raw
        files.append(('data', image_bytes))
    response = requests.post(url=url, files=files)
    response = response.json() 
    print(response)
    # assert response.status_code == 201, f"Request on {url} failed"
    assert response['feature'] is not None
    return response['feature']

if __name__ == "__main__":
    # test_encode_single_text()
    # test_encode_batch_text()
    x1=test_encode_single_image()['data']
    x2=test_encode_image_url()['data']
    x3=test_encode_batch_image()['data']
    import pdb; pdb.set_trace()
