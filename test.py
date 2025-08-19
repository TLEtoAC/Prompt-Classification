import os
os.environ["USE_TF"] = "0"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score

# 1. Load dataset from JSON
dataset = load_dataset("json", data_files="c:/Users/LENOVO/Desktop/EDI/sample classification of prompts/prompt_tier_dataset.json")

# 2. Encode labels
label_list = list(set(dataset["train"]["tier"]))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# 3. Tokenizer - Use a proper classification model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    return tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(preprocess, batched=True)
dataset = dataset.rename_column("tier", "labels")
# Fix: Convert string labels to integers
dataset = dataset.map(lambda x: {"labels": label2id[x["labels"]]}, batched=False)

# 4. Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./tier_classifier",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer
)

# 7. Train
trainer.train()

# 8. Save model
trainer.save_model("./tier_classifier")
tokenizer.save_pretrained("./tier_classifier")

# 9. Evaluate accuracy
predictions = trainer.predict(dataset["train"])
pred_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1)
true_labels = torch.tensor(predictions.label_ids)
accuracy = (pred_labels == true_labels).float().mean().item()
print(f"Training Accuracy: {accuracy:.4f}")

# 10. Test prediction
test_prompt = "Summarize a 50-page legal contract in simple words."
inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
pred_label = id2label[torch.argmax(outputs.logits).item()]
print(f"Predicted Tier: {pred_label}")
