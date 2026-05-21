import numpy as np
import base64
from PIL import Image
from io import BytesIO

def generate_random_image():
    random_image = np.random.randint(0, 256, (28, 28, 3), dtype=np.uint8)
    return random_image

def encode_image(image_array, name='mnist', i=0):
    if name == 'mnist':
        pil_img = image_array.reshape(28, 28)
    else:
        pil_img = image_array
    return pil_img