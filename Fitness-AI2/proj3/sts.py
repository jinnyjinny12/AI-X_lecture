#step1
from sentence_transformers import SentenceTransformer

#step2
model = SentenceTransformer("paraphrase-multilingula-MiniLM-L12-v2")

#step3
sentence1 = "The weather is lovely today."
sentence2 = "It's so sunny outside!"
sentence3 = "He drove to the stadium."


#step4
# embeddings = model.encode(sentences)
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

print(embedding1.shape)

#step5
similarities = model.similarity(embedding1, embedding2)
print(similarities)
