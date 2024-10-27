# pip install "fastapi[standard]"

# STEP 1 : import modules
# import argparse
from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2 : create inference object(instance)
face = FaceAnalysis()
face.prepare(ctx_id=0, det_size=(640, 640))


app = FastAPI()


@app.post("/uploadfile/")
async def create_upload_file(file1: UploadFile, file2: UploadFile):
    contents1 = await file1.read()
    contents2 = await file2.read()

    # STEP 3 : load data
    binary1 = np.fromstring(contents1, dtype=np.uint8)
    binary2 = np.fromstring(contents2, dtype=np.uint8)

    img1 = cv2.imdecode(binary1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(binary2, cv2.IMREAD_COLOR)

    # STEP 4 : inference
    faces1 = face.get(img1)
    assert len(faces1) == 1

    faces2 = face.get(img2)
    assert len(faces2) == 1

    # STEP 5 : Post processing (application)
    feat1 = faces1[0].normed_embedding
    feat2 = faces2[0].normed_embedding

    np_feat1 = np.array(feat1, dtype=np.float32)
    np_feat2 = np.array(feat2, dtype=np.float32)

    sims = np.dot(np_feat1, np_feat2.T)
    print(sims)

    return {"similarity": sims.item()}
    # sims.item()
    # numpy -> convert -> python data

# fastapi dev api_insightFace_test.py
