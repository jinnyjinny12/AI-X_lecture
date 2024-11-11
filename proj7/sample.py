import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 1. 데이터 로드
with open("sample.json", "r") as file:
    data = json.load(file)

# 2. 모델 초기화
model = SentenceTransformer('all-MiniLM-L6-v2')  # SentenceTransformer 모델
gpt_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")  # Hugging Face에서 제공되는 모델

# 3. FAISS 인덱스 생성
dimension = 384  # all-MiniLM-L6-v2 벡터 차원
index = faiss.IndexFlatL2(dimension)

# 4. 데이터 벡터화 및 삽입
vectors = []
metadata = []

for item in data:
    vector = model.encode(item["content"]).astype("float32")
    vectors.append(vector)
    metadata.append(item)

# FAISS에 벡터 삽입
faiss_data = np.array(vectors)
index.add(faiss_data)

# 데이터 삽입 확인
print(f"Number of items in FAISS index: {index.ntotal}")

# 5. 검색 및 진단 생성 함수
def diagnose_user_input(user_input):
    # 사용자 입력 벡터화
    query_vector = model.encode(user_input).astype("float32").reshape(1, -1)
    
    # FAISS에서 유사 데이터 검색
    distances, indices = index.search(query_vector, k=1)
    matched_index = indices[0][0]
    matched_data = metadata[matched_index]

    # 진단 생성
    content = matched_data["content"]
    diagnosis = gpt_generator(
        f"User Input: {user_input}\nBased on this, here's some advice: {content}",
        max_length=100
    )[0]["generated_text"]
    
    return diagnosis

# 6. 콘솔 인터페이스
if __name__ == "__main__":
    print("Welcome to the Posture Diagnosis System!")
    while True:
        user_input = input("\nEnter your symptoms or posture description (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        result = diagnose_user_input(user_input)
        print(f"\nDiagnosis:\n{result}")
