# 이미지에서 감정 추출 (예: deepinsight/FER 모델 사용)
from transformers import pipeline

image_analyzer = pipeline("image-classification", model="deepinsight/FER")
image_path = "img/family.jpg"
emotion_result = image_analyzer(image_path)

# 감정 분석 결과를 텍스트 모델에 입력하여 자세한 진단 생성
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

emotion_text = emotion_result[0]['label']  # 예: '행복'
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-emotion")

inputs = tokenizer.encode(emotion_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=50)
diagnosis = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("진단 결과:", diagnosis)
