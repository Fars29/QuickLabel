import cv2
import numpy as np
from core.image_processor import letterbox
from PIL import Image

pil_img = Image.new('RGB', (640, 480), (100, 100, 100))
arr = np.array(pil_img)[:, :, ::-1].copy()

try:
    tracker = cv2.TrackerCSRT_create()
except:
    tracker = cv2.legacy.TrackerCSRT_create()

print('Contiguous:', arr.flags['C_CONTIGUOUS'])
print('dtype:', arr.dtype)
print('shape:', arr.shape)
ok = tracker.init(arr, (234, 65, 159, 313))
print('init:', ok)
