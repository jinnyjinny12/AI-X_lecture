
# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.112)
detector = vision.ObjectDetector.create_from_options(options)


from fastapi import FastAPI, UploadFile
app = FastAPI()

import cv2
import numpy as np

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    #step3
    binary = np.fromstring(contents, dtype = np.uint8)
    cv_mat = cv2.imdecode(binary, cv2.IMREAD_COLOR)
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

    # image = mp.Image.create_from_file(IMAGE_FILE)

   # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(rgb_frame)

   # STEP 5
    person_count = 0
    for detection in detection_result.detections:
        category = detection.categories[0]
        if category.category_name == 'person':
            person_count += 1
    # print(f'find : {person_count}')
        return {"result" : person_count}