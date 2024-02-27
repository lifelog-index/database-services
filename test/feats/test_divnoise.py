import requests
import rich
from dotenv import load_dotenv
from pathlib import Path 
import os 


dotenv_path = Path(os.path.realpath(__file__)).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

DIV_NOISE_PORT = os.environ.get("DIV_NOISE_PORT", None)

assert DIV_NOISE_PORT is not None, "DIV_NOISE_PORT is not set"

def test_divnoise():
    urls = [f'http://localhost:{DIV_NOISE_PORT}/api/extract_noiseprint',f'http://localhost:{DIV_NOISE_PORT}/api/extract_exif']
    for url in urls:
        img_path = Path(os.path.realpath(__file__)).parent / 'tmp.jpg'
        resp = requests.post(url=url, files=[('file', open(img_path, 'rb'))])
        rich.print(resp.json())

if __name__ == "__main__":
    test_divnoise()