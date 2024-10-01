# 1st APPROACH Load model directly
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

tweet = "this is great!"

inputs = tokenizer(tweet, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class = logits.argmax().item()

# Label mapping
labels = ["negative", "positive"]

# Print the predicted class
print(f"The predicted class for the tweet '{tweet}' is: {labels[predicted_class]}")

# 2nd APPROACH Use HF pipeline
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier(tweet)

print(result)