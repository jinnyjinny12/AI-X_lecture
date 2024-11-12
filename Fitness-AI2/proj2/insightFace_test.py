# https://github.com/deepinsight/insightface/blob/master/examples/demo_analysis.py
# https://onnxruntime.ai/ -> pip install onnxruntime

# STEP 1 : import modules
# import argparse
import cv2
# import sys
import numpy as np
# import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# assert insightface.__version__>='0.3'

# parser = argparse.ArgumentParser(description='insightface app test')
# # general
# parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
# parser.add_argument('--det-size', default=640, type=int, help='detection size')
# args = parser.parse_args()

# STEP 2 : create inference object(instance)
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3 : load data
# img = ins_get_image('t1')
img1 = cv2.imread('image\\iu1.jpg')
img2 = cv2.imread('image\\iu2.jpg')
# imread?
# file open
# decode img 

# STEP 4 : inference
faces1 = app.get(img1)
assert len(faces1)==1

faces2 = app.get(img2)
assert len(faces2)==1

# STEP 5 : Post processing (application)
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# then print all-to-all face similarity
feat1 = faces1[0].normed_embedding
feat2 = faces2[0].normed_embedding

np_feat1 = np.array(feat1, dtype=np.float32)
np_feat2 = np.array(feat2, dtype=np.float32)

# feats = []
# for face in faces:
#     feats.append(face.normed_embedding)

sims = np.dot(np_feat1, np_feat2.T)
print(sims)