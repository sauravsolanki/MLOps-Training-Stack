import base64

import requests


# decode_base64_img(encode_str)
def encode_image(path):
    with open(path, 'rb') as f:
        image_read = f.read()
    image_64_encode = base64.b64encode(image_read)
    return image_64_encode


image_base64_str = '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APn+tzw34O1/xdPLFoemyXZiGZGDKirnplmIGfbNdOPgp4zjheW9t7CxVOv2m+iHHrkEj864/XtDuPD2pmwuri0nlCBy1pOsqDPbI7+1Zg5Ne5+OrbW/BvgPTNA8JWl22jyW32nUNYslLLcOww2WXOxfqRkEDsa8PlmlnkLzSPI/Tc7En9aZU1okEl5BHdTNDbtIollVNxRSeWA4zgc4r2Xwt4R+KOga3FBoOqBtGyrrem4V7N4j/GEJ547AZ6c9K5D4wTaDP8Qbp9AMLR+Wv2uS3/1b3GTvZeSP7ucd8/WuDoqxHf3kVs1tHdzpbt96JZCFP1GcVXor/9k='

payload = {
    "data": image_base64_str,
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=payload)
print(response.json())
