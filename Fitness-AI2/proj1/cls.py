# Env download

# models download - EfficientNet-Lite0 (float 32)
# https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite

# Image Classifier
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/image_classification/python/image_classifier.ipynb?hl=ko#scrollTo=gxbHBsF-8Y_l

# Download test images

# import urllib.request

IMAGE_FILENAMES = ['image/burger.jpg', 'image/cat.jpg']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}'
#   urllib.request.urlretrieve(url, name)

import cv2 # opencv lib
# from google.colab.patches import cv2_imshow
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
#   cv2_imshow(img)
  cv2.imshow("test", img)
  cv2.waitKey(0)


# Preview the images.
# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)

# Running inference and visualizing the results

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

# STEP 3: Load the input image.
image = mp.Image.create_from_file('burger.jpg')

# STEP 4: Classify the input image.
classification_result = classifier.classify(image)
# print(classification_result)

# [example]
# ClassificationResult(
#   classifications=[
#     Classifications(
#       categories=[
#         Category(index=933, score=0.9790240526199341, display_name='', category_name='cheeseburger'), 
#         Category(index=931, score=0.0008637145510874689, display_name='', category_name='bagel'), 
#         Category(index=947, score=0.0005722717614844441, display_name='', category_name='mushroom')
#       ], 
#       head_index=0, 
#       head_name='probability'
#     )
#   ], 
#   timestamp_ms=0
# )

# STEP 5: Process the classification result. In this case, visualize it.
top_category = classification_result.classifications[0].categories[0]
print(f"{top_category.category_name} ({top_category.score:.2f})")