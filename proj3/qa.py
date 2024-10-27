#step1 : import module
from transformers import pipeline

#step2 : create inference object(instance)
question_answerer = pipeline("question-answering", model="stevhliu/my_awesome_qa_model")

#step3 : prepare data
question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

#step4 : inference
result = question_answerer(question=question, context=context)


#step5 : post processing
print(result)

