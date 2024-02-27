from io import BytesIO
import base64
from PIL import Image

def openImageb64(b64_data):
    im_bytes = base64.b64decode(b64_data)
    im_file = BytesIO(im_bytes)  
    image = Image.open(im_file)
    return image
