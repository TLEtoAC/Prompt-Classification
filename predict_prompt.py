# predict_prompt.py
import sys, json, joblib
from sentence_transformers import SentenceTransformer

# Load model + embedder
svm_model = joblib.load("prompt_quality_model.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Read input
data = json.loads(sys.argv[1])
prompt = data["prompt"]

# Convert to embeddings     
embedding = embedder.encode([prompt])

# Predict
prediction = svm_model.predict(embedding)
print(prediction[0])
