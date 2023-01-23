#!/usr/bin/env python
# coding: utf-8
import base64
import json
import os
from abc import abstractmethod
from io import BytesIO
from typing import Dict

import mlflow
import numpy as np
import requests
from PIL import Image


# url = 'http://bit.ly/mlbookcamp-pants'


class InferenceEngine:
    def __init__(self):
        self.RUN_ID = os.getenv('RUN_ID')
        # self.ROOT_PATH_MODEL = os.getenv('ROOT_PATH_MODEL', './')

        assert self.RUN_ID is not None, "NO RUN_ID model found"
        # assert self.ROOT_PATH_MODEL is not None, "NO BASE_PATH model found"

        print("Model Loading..")

        self.logged_model = f'mlruns/1/{self.RUN_ID}/artifacts/keras-model'
        print(self.logged_model)

        self.model = mlflow.pyfunc.load_model(self.logged_model)
        self.class_names = []

        print("Model Loaded")

    @abstractmethod
    def predict(self, image):
        pass

    @abstractmethod
    def parse_payload(self, payload: Dict):
        pass

    @abstractmethod
    def __decode(self, image_64_encode):
        pass


class LambdaPreprocessor(InferenceEngine):
    def __init__(self):
        super().__init__()

    def predict(self, image):
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)
        pred = self.model.predict(image)
        class_names = np.argmax(pred[0])
        return list(range(10)), pred[0]

    def parse_payload(self, payload: Dict):
        return self.__decode(payload)

    def __decode(self, payload):
        if 'url' in payload:
            response = requests.get(payload['url'])
            img = Image.open(BytesIO(response.content))
            return img
        elif 'base64' in payload:
            image_64_encode = payload['base64']
            image_64_decode = base64.b64decode(image_64_encode)
            image = Image.open(BytesIO(image_64_decode))
            return np.array(image)
        else:
            raise NotImplementedError(f"{payload.keys()} can not be parsed")


preprocessor = LambdaPreprocessor()


def predict(payload):
    img = preprocessor.parse_payload(payload)
    class_name, prods = preprocessor.predict(img)
    return dict(zip(class_name, prods))


def lambda_handler(event, context):
    result = predict(event)
    return json.dumps(str(result))
    # return str(result)


if __name__ == '__main__':
    image_base64_str = '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APn+tzw34O1/xdPLFoemyXZiGZGDKirnplmIGfbNdOPgp4zjheW9t7CxVOv2m+iHHrkEj864/XtDuPD2pmwuri0nlCBy1pOsqDPbI7+1Zg5Ne5+OrbW/BvgPTNA8JWl22jyW32nUNYslLLcOww2WXOxfqRkEDsa8PlmlnkLzSPI/Tc7En9aZU1okEl5BHdTNDbtIollVNxRSeWA4zgc4r2Xwt4R+KOga3FBoOqBtGyrrem4V7N4j/GEJ547AZ6c9K5D4wTaDP8Qbp9AMLR+Wv2uS3/1b3GTvZeSP7ucd8/WuDoqxHf3kVs1tHdzpbt96JZCFP1GcVXor/9k='
    out = lambda_handler({"base64": image_base64_str}, {})
    print(out)
