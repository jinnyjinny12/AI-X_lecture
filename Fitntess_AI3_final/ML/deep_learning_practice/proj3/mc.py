

#step 1
from transformers import AutoModelForMultipleChoice
from transformers import AutoTokenizer
import torch

#step 2
tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_swag_model")
model = AutoModelForMultipleChoice.from_pretrained("stevhliu/my_awesome_swag_model")
labels = torch.tensor(0).unsqueeze(0)

#step 3
prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
candidate1 = "The law does not apply to croissants and brioche."
candidate2 = "The law applies to bagettes."


#step 4
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True) #전처리
outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
logits = outputs.logits
predicted_class = logits.argmax().item()

#step 5
print(predicted_class)


# data -> 래핑된 모델 -> 추론기 추론 -> 













