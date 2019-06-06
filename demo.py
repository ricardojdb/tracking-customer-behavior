from io import BytesIO
import numpy as np
import requests
import base64
import cv2

import utils


def encode_img(img):
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)
    return img_str

# Start thread to capture and show the stream.
video_path = 0
video_capture = utils.WebcamVideoStream(video_path).start()

host = "localhost"

while True:
    # Collect width and height from the stream
    h, w = int(video_capture.h), int(video_capture.w)
    # Read the current frame
    ret, image = video_capture.read()

    if not ret:
        print('No frames has been grabbed')
        break

    img = np.copy(image)

    try:
        detect_req = requests.post(
            url=f'http://{host}:7000/predict',
            data=encode_img(img),
            timeout=5)
        detections = detect_req.json()
    except:
        traceback.print_exc(file=sys.stdout)
        detections = []

    data_list = []
    for detection in detections:
        label = detection["label"]
        xmin = int(detection["xmin"] * w)
        ymin = int(detection["ymin"] * h)
        xmax = int(detection["xmax"] * w)
        ymax = int(detection["ymax"] * h)

        data_list.append([label, [xmin, ymin, xmax, ymax]])

    # Send outputs to the thread so it can be plotted on the stream.
    video_capture.data_list = data_list

    if video_capture.stopped:
        break
