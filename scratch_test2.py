import cv2
import numpy as np
from core.image_processor import letterbox
from PIL import Image

pil_img = Image.new('RGB', (640, 480), (100, 100, 100))
arr = np.array(pil_img)[:, :, ::-1].copy()

tracker = cv2.TrackerCSRT_create()
ok = tracker.init(arr, (234, 65, 159, 313))
print('init ok:', ok)

res = tracker.update(arr)
print('update res:', res)
