
#step 1 : import module
from transformers import pipeline
#step 2 : create inference object
question_answerer = pipeline("question-answering", model="stevhliu/my_awesome_qa_model")
#step 3 : prepare data
question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."


# step 4 : ingerence
result = question_answerer(question=question, context=context)
# step 5: post processing
print(result)




