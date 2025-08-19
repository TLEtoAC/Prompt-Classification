test_prompt = "Summarize a 50-page legal contract in simple words."
inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
pred_label = id2label[torch.argmax(outputs.logits).item()]
print(f"Predicted Tier: {pred_label}")
