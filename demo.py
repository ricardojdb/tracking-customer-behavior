from io import BytesIO
import numpy as np
import requests
import base64
import cv2


def encode_img(img):
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)
    return img_str

img = cv2.imread("image.jpg")
img = img[:, :, ::-1]
print(img.shape)
h, w = img.shape[:2]
img_str = encode_img(img)

host = "localhost"
r = requests.post(f"http://{host}:7000/predict", data=img_str)

detections = r.json()[0]
xmin = int(detections["xmin"] * w)
ymin = int(detections["ymin"] * h)
xmax = int(detections["xmax"] * w)
ymax = int(detections["ymax"] * h)

img_out = img[:, :, ::-1]
img_out = cv2.rectangle(img_out, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
cv2.imshow("image", img_out)
cv2.waitKey()
cv2.destroyAllWindows()
