import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Setup
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Check if MPS is available and set the device accordingly
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

# Prepare datasets
product_data = {
    "text": [
        "The new smartphone features a high-resolution camera and long battery life.",
        "Climate change is affecting global weather patterns and ecosystems.",
        "This moisturizer leaves your skin feeling soft and hydrated all day long.",
        "The election results will be announced tonight on all major news channels.",
        "Our vacuum cleaner effectively removes pet hair from carpets and upholstery.",
        "Scientists have discovered a new exoplanet orbiting a distant star.",
        "The latest gaming console offers stunning graphics and immersive gameplay.",
        "Renewable energy sources are becoming increasingly important for sustainability.",
        "This ergonomic office chair provides excellent lumbar support for long work hours.",
        "The annual music festival will feature performances from top international artists.",
        "Our non-stick cookware set is dishwasher safe and comes with a lifetime warranty.",
        "The Great Barrier Reef is experiencing significant coral bleaching due to rising ocean temperatures.",
        "This smart home security system includes motion sensors and 24/7 monitoring.",
        "The United Nations is working on initiatives to promote global peace and cooperation.",
        "Our noise-cancelling headphones provide crystal-clear audio and comfortable wear.",
        "Researchers have made a breakthrough in understanding the human genome.",
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for product-related, 0 for other information
}

product_dataset = Dataset.from_dict(product_data)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_product = product_dataset.map(tokenize_function, batched=True)

# Evaluation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Classification function
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=-1).item()
    return prediction

# Test texts
test_texts = [
    "This smartphone has an amazing camera and long battery life.",
    "The latest climate report shows alarming trends in global temperatures.",
    "Our new blender can crush ice and make smoothies in seconds.",
    "The upcoming election will determine the country's economic policies.",
    "This moisturizer contains hyaluronic acid for deep hydration.",
    "Scientists have discovered a new species of deep-sea fish.",
    "The ergonomic keyboard helps reduce wrist strain during long typing sessions.",
    "Renewable energy sources are becoming more efficient and cost-effective."
]

# Fine-tune for new task
model.num_labels = 2
model.config.id2label = {0: "other", 1: "product"}
model.config.label2id = {"other": 0, "product": 1}
model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
model = model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_product,
    eval_dataset=tokenized_product,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate fine-tuned model on new task
results = trainer.evaluate()
print("\nFine-tuned model results on product-related task:", results)

print("\nClassifications after fine-tuning (0: Other, 1: Product-related):")
for text in test_texts:
    print(f"'{text}': {classify_text(text)}")
