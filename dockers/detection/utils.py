from io import BytesIO
from PIL import Image
import numpy as np
import base64
import json
import os

from models.yolov3 import Darknet, load_darknet_weights, non_max_suppression
from models.yolov3 import utils
import torch


class YoloDetection(object):
    """Handles data preprocess and forward pass of the model"""
    def __init__(self, model_path="models/yolov3/"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.img_size = (416, 416)
        self.classes = utils.load_classes(
            os.path.join(self.model_path, "coco.names"))
        self.model = self.init_model()

    def init_model(self):
        """Initializes the machine learning model.

        Returns:
            model (object): Loaded pre-trained model used
                to make predictions.

        """
        model_name = "yolov3"
        model = Darknet(
            cfg=os.path.join(self.model_path, f"cfg/{model_name}.cfg"),
            img_size=self.img_size)
        load_darknet_weights(
            model,
            os.path.join(
                self.model_path,
                f"weights/{model_name}.weights"))
        model.fuse()
        model.eval()
        model.to(self.device)

        return model

    def decode_data(self, encoded_data):
        """Decodes the encoded data comming from a request.

        Args:
            encoded_data (base64): data comming from the HTTP request.

        Returns:
            array: Data decoded into a usable format.

        """
        # NOTE: This could vary depending on your data
        return Image.open(BytesIO(base64.b64decode(encoded_data)))

    def preprocess(self, raw_data):
        """Prerocess the data into the right format
        to be feed in to the given model.

        Args:
            raw_data (array): Raw decoded data to be processed.

        Returns:
            array: The data ready to use in the given model.

        """

        img = raw_data.resize(self.img_size)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        return img

    def model_predict(self, encoded_data):
        """Decodes and preprocess the data, uses the
        pretrained model to make predictions and
        returns a well formatted json output.

        Args
            encoded_data (base64): data comming from the HTTP request.

        Returns:
            json: A response that contains the output from
                the pre-trained model.
        """
        # Decode data
        data = self.decode_data(encoded_data)
        # Preprocess into the right format
        inputs = self.preprocess(data)
        # Compute predictions
        pred, _ = self.model(inputs)
        detections = non_max_suppression(pred)[0]

        # Create json response
        detect_list = []
        if detections is None:
            return json.dumps(detect_list)

        for i in range(0, detections.shape[0]):
            # extract the confidence (i.e., probability
            # associated with the prediction
            box = detections[i, :4]
            label = self.classes[int(detections[i, -1])]

            (xmin, ymin, xmax, ymax) = box / self.img_size[0]

            face_json = {'label': label,
                         'xmin': float(xmin),
                         'ymin': float(ymin),
                         'xmax': float(xmax),
                         'ymax': float(ymax)}

            detect_list.append(face_json)

        return json.dumps(detect_list)
