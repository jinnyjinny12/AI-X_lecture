from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
IMAGE_FILE = 'images/people.jpg'
ANIMAL_FILE = 'images/animal.jpeg'


base_options = python.BaseOptions(model_asset_path='models/efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                    score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)


app = FastAPI()




@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()

    binary = np.fromstring(contents, dtype=np.uint8)
    cv_mat = cv2.imdecode(binary, cv2.IMREAD_COLOR)
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat) # 모델로 읽을 수 있게 이미지를 다시 만듬
    # image = mp.Image.create_from_file(rgb_frame)

    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(rgb_frame)

    # SETP 5:
    people_len = findPerson(detection_result)

    return {"result": people_len}



# STEP 1: Import the necessary modules.


# STEP 2: Create an ObjectDetector object.


def findPerson(detection_result):
    people_len = 0

    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
    
        if(category_name == 'person'):
            people_len += 1
        
    return people_len