
from fastapi import FastAPI, File, UploadFile
import easyocr

reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory

app = FastAPI()



@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

result = reader.readtext('chinese.jpg')