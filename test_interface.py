import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load trained model
model_path = "./tier_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_tier(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits).item()
    return model.config.id2label[pred_id]

# Test interface
test_prompts = [
    "What is the capital of France?",
    "Explain how photosynthesis works in 3 steps.",
    "Analyze the financial performance of Tesla in Q4 2024.",
    "Summarize a 50-page legal contract in simple words.",
    "Translate 'hello' to Spanish."
]

print("Prompt Tier Classification Results:")
print("-" * 50)
for prompt in test_prompts:
    tier = predict_tier(prompt)
    print(f"Prompt: {prompt}")
    print(f"Predicted Tier: {tier}")
    print("-" * 50)

# Interactive mode
print("\nInteractive Mode (type 'quit' to exit):")
while True:
    user_prompt = input("\nEnter a prompt: ")
    if user_prompt.lower() == 'quit':
        break
    tier = predict_tier(user_prompt)
    print(f"Predicted Tier: {tier}")