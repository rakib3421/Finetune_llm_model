from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the model and tokenizer
model_path = "stablelm-2-zephyr-1_6b-eade-finetuned/"
base_model_name = "stabilityai/stablelm-2-zephyr-1_6b"

print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="cpu"
)
model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load knowledge base and create embeddings
print("Loading knowledge base...")
with open('data.json', 'r') as f:
    knowledge_base = json.load(f)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
prompts = [item['prompt'] for item in knowledge_base]
responses = [item['response'] for item in knowledge_base]
documents = responses  # For compatibility with chatbot_response

print("Creating embeddings...")
embeddings = embedder.encode(prompts, convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def chatbot_response(input_text, top_k=3):
    try:
        # Retrieve relevant documents
        query_embedding = embedder.encode([input_text], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_k)
        
        # Select the most relevant document (smallest distance)
        min_distance_idx = indices[0][0]  # First index (most relevant)
        most_relevant_doc = documents[min_distance_idx]
        
        # Split document into sentences and select the first relevant sentence
        sentences = most_relevant_doc.split('. ')
        for sentence in sentences:
            if input_text.lower().replace("?", "").strip() in sentence.lower():
                return sentence.strip() + ('.' if not sentence.endswith('.') else '')
        
        # Fallback: return the first sentence of the most relevant document
        return sentences[0].strip() + ('.' if not sentences[0].endswith('.') else '')
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Please provide a message'}), 400
    
    user_message = data['message']
    
    # Generate response using retrieval
    response = chatbot_response(user_message)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
