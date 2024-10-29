from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# FastAPI 인스턴스 생성
app = FastAPI()

# 감정 분석 파이프라인 설정
sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

# 요청 데이터 모델 정의
class SentimentRequest(BaseModel):
    description: str  # 설명 필드 추가

# 감정 분석 엔드포인트 정의
@app.post("/analyze-sentiment/")
async def analyze_sentiment(request: SentimentRequest):
    try:
        # 입력 텍스트 감정 분석
        result = sentiment_analysis(request.description)
        return {"description": request.description, "sentiment": result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 실행을 위한 메인 구문
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
