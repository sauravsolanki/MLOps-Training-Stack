import io
import os
import base64
from typing import Dict

import numpy as np
import mlflow
from PIL import Image
from flask import Flask, jsonify, request


class InferenceEngine:
    def __init__(self):
        self.RUN_ID = os.getenv('RUN_ID')
        assert self.RUN_ID is not None, "NO RUN_ID model fpund"
        self.logged_model = f'/app/mlflow/mlruns/1/{self.RUN_ID}/artifacts/keras-model'
        self.model = mlflow.pyfunc.load_model(self.logged_model)
        self.class_names = []

    def predict(self, image):
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)
        pred = self.model.predict(image)
        return float(np.argmax(pred[0]))
        # return self.class_names[np.argmax(pred[0])]

    def parse_payload(self, payload: Dict):
        return self.__decode_base64_img(payload["data"])

    def __decode_base64_img(self, image_64_encode):
        image_64_decode = base64.b64decode(image_64_encode)
        image = Image.open(io.BytesIO(image_64_decode))
        image_np = np.array(image)
        return image_np


app = Flask('predict-mnist')
ie = InferenceEngine()


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    image = ie.parse_payload(request.get_json())
    print(image.shape)

    pred = ie.predict(image)

    result = {'duration': pred, 'model_version': ie.RUN_ID}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
