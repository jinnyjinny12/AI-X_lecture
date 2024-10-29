from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import io

app = FastAPI()

# 이미지 캡셔닝 모델과 감정 분석 모델 초기화
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
emotion_classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

# 요청 데이터 모델 정의
class ImageRequest(BaseModel):
    url: str

# 이미지 감정 분석 엔드포인트
@app.post("/analyze-image-sentiment/")
async def analyze_image_sentiment(request: ImageRequest):
    try:
        # 이미지 로드
        response = requests.get(request.url)
        image = Image.open(io.BytesIO(response.content))
        
        # 이미지 캡셔닝
        inputs = caption_processor(images=image, return_tensors="pt")
        caption_ids = caption_model.generate(**inputs)
        caption = caption_processor.decode(caption_ids[0], skip_special_tokens=True)

        # 텍스트 캡션으로 감정 분석 수행
        emotion_result = emotion_classifier(caption)

        return {"caption": caption, "emotion": emotion_result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI 실행을 위한 메인 구문
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
