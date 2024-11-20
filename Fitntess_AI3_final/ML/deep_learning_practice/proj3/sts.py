#step 1
from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
#step 2
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# step 3
sentence1 = "집에 갑시다."
sentence2 = "안녕하세요."
sentence3 = "He drove to the stadium."
# The sentences to encode


# 2. Calculate embeddings by calling model.encode()

#step 4
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
embedding3 = model.encode(sentence3)
#print(embedding2.shape)

#step 5
similarities = model.similarity(embedding1, embedding2)
print(similarities)
