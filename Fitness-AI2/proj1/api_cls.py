
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision



# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='models\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)


from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# import cv2
# import numpy as np
# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile):
#     contents = await file.read()
#     # STEP 3: Load the input image.
#     binary = np.fromstring(contents, dtype = np.uint8)  
#     cv_mat = cv2.imdecode(binary, cv2.IMREAD_COLOR)
#     rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
#     # image = mp.Image.create_from_file('burger.jpg')
#     # STEP 4: Classify the input image.
#     classification_result = classifier.classify(rgb_frame)
#     # print(classification_result)
#     # STEP 5: Process the classification result. In this case, visualize it.
#     top_category = classification_result.classifications[0].categories[0]
#     # print(f"{top_category.category_name} ({top_category.score:.2f})")
#     return {"category": top_category.category_name,
#             "score": f"{top_category.score:.2f}"}

# 한번더   content 파일로 받아주자
import cv2
import numpy as np
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()

    # STEP 3: Load the input image.
    binary = np.fromstring(contents, dtype = np.uint8)
    cv_mat = cv2.imdecode(binary, cv2.IMREAD_COLOR)
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
    # image = mp.Image.create_from_file('burger.jpg')

    # STEP 4: Classify the input image.
    classification_result = classifier.classify(rgb_frame)
    # print(classification_result)

    # STEP 5: Process the classification result. In this case, visualize it.
    top_category = classification_result.classifications[0].categories[0]
    # print(f"{top_category.category_name} ({top_category.score:.2f})")

    return {"category": top_category.category_name,
            "score": f"{top_category.score:.2f}"}

