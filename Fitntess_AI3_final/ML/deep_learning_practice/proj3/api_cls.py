from fastapi import FastAPI, Form
#sep 1 : import module
from transformers import pipeline

# step 2 : create inference object(instance)
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

app = FastAPI()


@app.post("/inference/")
async def login(text: str = Form()):
    # step 3 : prepare data
    # text = "LG전자, 獨서 미래 모빌리티 핵심 기술 'Soft V2X' 공개…""자율주행 리더십 확보할 것"""

    # step 4 : inference
    result = classifier(text)

    # step 5 : post processing
    # print(result)
    # label = 0 부정 1은 긍정'
    print(result)
    #natual 중립
    if(result[0]['label'] == 'positive'):
        return {"result": result}
    else:
        return {"result": 'null'}