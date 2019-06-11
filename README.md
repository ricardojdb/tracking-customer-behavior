# Tracking Customer Behavior System

A people tracking system using YOLOv3 to analyze customer behavior inside an establishment.

## Installation

Install the following packages

```bash
pip install pillow numpy requests
pip install opencv-python
pip install moviepy
```

## Usage

First you have to start the docker APIs

```bash
docker-compose dockers\docker-compose.yml build
docker-compose dockers\docker-compose.yml up -d
```

Than you can test the API by running the `demo.py` file.
```bash
python demo.py
```

To call the API use the following.

An HTTP Post request with an image as `data` in `base64`

URL:
```bash
http://0.0.0.0:7000/predict
```

Python example using OpenCV:
```python
import requests
import base64
import cv2

# Read the example image
img = cv2.imread("image.jpg")

# Transform the image to base64
_, buffer = cv2.imencode('.jpg', img)
img_str = base64.b64encode(buffer)

# POST request to a Localhost (this can be changed for public or private ip)
req = requests.post('http://localhost:7000/predict', data=img_str)

# Print results
print(req.json())
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
