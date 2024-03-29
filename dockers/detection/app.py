from flask import Flask, request
import numpy as np
import utils

# Initialize the flask app
app = Flask(__name__)

# Loads the given model
model = utils.YoloDetection("models/yolov3/")


@app.route('/predict', methods=['POST'])
def predict():
    # Obtain the data from the request
    data = request.get_data()
    # Runs the model and returns the outputs in a json format
    output = model.model_predict(data)
    return output

if __name__ == "__main__":
    # Running the Flask app on the url http://0.0.0.0:7000/
    # Use 0.0.0.0 to run in any IP available
    app.run(host='0.0.0.0', port=7000)
