from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

# FastAPI 인스턴스 생성
app = FastAPI()

# CLIP 모델과 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 요청 데이터 모델
class ImageTextRequest(BaseModel):
    url: str
    texts: list

# 엔드포인트 정의
@app.post("/predict/")
async def predict(request: ImageTextRequest):
    # 이미지 로드
    try:
        image = Image.open(requests.get(request.url, stream=True).raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image URL")

    # 텍스트와 이미지를 처리하여 모델에 입력
    inputs = processor(text=request.texts, images=image, return_tensors="pt", padding=True)

    # 예측 실행
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).tolist()

    # 결과 반환
    return {"similarity_scores": probs}
