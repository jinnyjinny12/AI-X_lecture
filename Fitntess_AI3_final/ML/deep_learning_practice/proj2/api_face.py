from fastapi import FastAPI, File, UploadFile
import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

face = FaceAnalysis()
face.prepare(ctx_id=0, det_size=(640,640))

app = FastAPI()



@app.post("/uploadfile/")
async def create_upload_file(file1: UploadFile, file2: UploadFile):
    contents1 = await file1.read()
    contents2 = await file2.read()

    binary1 = np.fromstring(contents1, np.uint8)
    binary2 = np.fromstring(contents2, np.uint8)

    # decode img 일반 비트맵으로 디코딩
    img1 = cv2.imdecode(binary1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(binary2, cv2.IMREAD_COLOR)
    # file open
   

    # 스텝 4 : inference
    faces1 = face.get(img1)
    assert len(faces1)==1

    faces2 = face.get(img2)
    assert len(faces2)==1


    # then print all-to-all face similarity
    # face.normed_embedding 
    emb1 = faces1[0].normed_embedding
    emb2 = faces2[0].normed_embedding

    np_emb1 = np.array(emb1, dtype=np.float32)
    np_emb2 = np.array(emb2, dtype=np.float32)


    # feats.append(face.normed_embedding)
    # feats = np.array(feats, dtype=np.float32) #numpy 어레이로 바꿔줌 dot을 쓰기위해


    # 스텝 5 : Post procexssing (application)
    # sims = np.dot(feats, feats.T) # 행렬연산 코사인시뮬럴리티\

    sims = np.dot(np_emb1, np_emb2)
    
    return {"filename": sims.item()}