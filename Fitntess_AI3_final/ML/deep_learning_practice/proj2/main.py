# 스텝 1 : import modules
import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# assert insightface.__version__>='0.3'


# parser = argparse.ArgumentParser(description='insightface app test')
# general
# parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
# parser.add_argument('--det-size', default=640, type=int, help='detection size')
# args = parser.parse_args()


# 추론기 onnxruntime 
# 설치를 해야함. pip install onnxruntime

#스텝 2 : create inference object(instance)
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))


# 스텝 3 : load data
# img = ins_get_image('t1')
img1 = cv2.imread('image/iu1.jpeg')
img2 = cv2.imread('image/iu_f2.png')
# file open
# decode img 일반 비트맵으로 디코딩




# 스텝 4 : inference
faces1 = app.get(img1)
assert len(faces1)==1

faces2 = app.get(img2)
assert len(faces2)==1

# 스텝 5 : Post procexssing (application)
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# then print all-to-all face similarity
# face.normed_embedding 
emb1 = faces1[0].normed_embedding
emb2 = faces2[0].normed_embedding

# feats = []
# for face in faces:

np_emb1 = np.array(emb1, dtype=np.float32)
np_emb2 = np.array(emb2, dtype=np.float32)


# feats.append(face.normed_embedding)
# feats = np.array(feats, dtype=np.float32) #numpy 어레이로 바꿔줌 dot을 쓰기위해


# sims = np.dot(feats, feats.T) # 행렬연산 코사인시뮬럴리티\

sims = np.dot(np_emb1, np_emb2)
print(sims)
# 0.4 미만이 다른 사람
