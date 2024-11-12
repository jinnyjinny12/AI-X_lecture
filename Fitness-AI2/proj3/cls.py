#step1: import modules
from transformers import pipeline

#step2: create inference object(instance)
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

#step3: prepare data
text = "마트 가면 안다, 사과·배추에 커피까지 무섭다는 걸”…서민 잡는 ‘기후플레이션’"

#step4: inference
result=classifier(text)

#step5: inference
print(result)
