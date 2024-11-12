#STEP1 : IMPORT MODULES
import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


#STEP2 : create inference object(instance)
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

#STEP3: load data
# img = ins_get_image('t1')
img1 = cv2.imread('iu.jpg')
img2 = cv2.imread('noiu5.jpg')

#file open
#decode img

#STEP4: inference (추론)
faces1 = app.get(img1)
assert len(faces1)==1

faces2 = app.get(img2)
assert len(faces2)==1


#STEP5 : Post Processing (application) 
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# then print all-to-all face similarity

emb1 = faces1[0].normed_embedding
emb2 = faces2[0].normed_embedding

# feats = []
# for face in faces:
#     feats.append(face.normed_embedding)

np_emb1 = np.array(emb1, dtype=np.float32)
np_emb2 = np.array(emb2, dtype=np.float32)


sims = np.dot(emb1, emb2.T)
print(sims)

