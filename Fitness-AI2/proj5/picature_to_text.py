from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

app = FastAPI()

# 모델과 프로세서 초기화
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

@app.post("/caption/")
async def generate_caption(file: UploadFile = File(...)):
    try:
        # 파일을 이미지로 열기
        image = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 이미지 캡션 생성
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(inputs["pixel_values"], max_length=100, num_return_sequences=1, num_beams=1)
    captions = [processor.decode(output, skip_special_tokens=True)[:100] for output in outputs]

    return {"captions": captions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main1:app", host="0.0.0.0", port=8000, reload=True)
