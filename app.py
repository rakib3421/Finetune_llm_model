import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import re
from llama_cpp import Llama

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

# Load StableLM model
print("Loading StableLM model...")
llm = Llama(
    model_path="stablelm-2-zephyr-1.6B-finetuned-F16.gguf",
    n_ctx=4096,
    n_threads=4,
    verbose=False
)
print("Model loaded successfully!")

# Initialize RAG components
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
prompts = [item['prompt'] for item in data]
responses = [item['response'] for item in data]

# Embed prompts (we will retrieve by prompt similarity and return the stored response)
prompt_embeddings = retriever_model.encode(prompts, convert_to_numpy=True)

# Create FAISS index on prompts
dimension = prompt_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(prompt_embeddings)

# Note: for short, specific answers we use retrieval-only (return the stored response).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Conversation context storage
conversation_contexts = {}

def detect_greeting(text):
    """Detect if the input is a greeting"""
    greetings = [
        r'\b(hello|hi|hey|greetings|good morning|good afternoon|good evening|how are you)\b',
        r'^\s*(hello|hi|hey)\s*$',
        r'good\s+(morning|afternoon|evening|day)',
        r'how\s+(are\s+you|do\s+you\s+do)'
    ]
    
    text_lower = text.lower().strip()
    for pattern in greetings:
        if re.search(pattern, text_lower):
            return True
    return False

def detect_farewell(text):
    """Detect if the input is a farewell"""
    farewells = [
        r'\b(goodbye|bye|farewell|see you|take care|thanks|thank you)\b',
        r'^\s*(bye|goodbye)\s*$',
        r'thank\s+you',
        r'thanks'
    ]
    
    text_lower = text.lower().strip()
    for pattern in farewells:
        if re.search(pattern, text_lower):
            return True
    return False

def get_contextual_response(input_text, session_id, conversation_history):
    """Generate response considering conversation context"""
    
    # Check if it's a greeting
    if detect_greeting(input_text):
        # Check if it's their first message
        if not conversation_history:
            return get_greeting_response(input_text)
        else:
            return "Hello again! How can I help you with EADE today?"
    
    # Check if it's a farewell
    if detect_farewell(input_text):
        return get_farewell_response(input_text)
    
    # Check for follow-up questions
    if len(conversation_history) > 0:
        last_response = conversation_history[-1].get('response', '')
        if 'program' in last_response.lower() and any(word in input_text.lower() for word in ['tell me more', 'details', 'more info', 'what about', 'how about']):
            return "Sure! Which program interests you? We offer programs in Business, Marketing, Engineering, and Psychology."
    
    # If the question seems too general or unclear, provide helpful guidance
    if len(input_text.split()) < 3 and not any(word in input_text.lower() for word in ['what', 'how', 'when', 'where', 'why', 'who']):
        return "Please be more specific about what you'd like to know about EADE. I can help with programs, admissions, or campus info."
    
    return None

def get_greeting_response(input_text):
    """Get appropriate greeting response"""
    current_hour = datetime.now().hour
    
    if 'morning' in input_text.lower() or (5 <= current_hour < 12):
        return "Good morning! I'm here to help with information about EADE Business School. How can I assist you?"
    elif 'afternoon' in input_text.lower() or (12 <= current_hour < 17):
        return "Good afternoon! Welcome to EADE Business School. What can I help you with today?"
    elif 'evening' in input_text.lower() or (17 <= current_hour < 22):
        return "Good evening! I'm here to answer your questions about EADE. What would you like to know?"
    else:
        return "Hello! I'm your EADE Business School virtual assistant. How can I help you?"

def get_farewell_response(input_text):
    """Get appropriate farewell response"""
    if 'thank' in input_text.lower():
        return "You're welcome! Glad I could help. Feel free to ask if you have more questions about EADE."
    else:
        return "Goodbye! Thank you for your interest in EADE Business School. Have a great day!"

def chatbot_response(input_text, session_id=None, top_k=1, similarity_threshold=None):
    """Enhanced chatbot response with context awareness and receptionist behavior"""
    try:
        # Get or create conversation history for this session
        if session_id not in conversation_contexts:
            conversation_contexts[session_id] = []
        
        conversation_history = conversation_contexts[session_id]
        
        # Check for contextual responses first
        contextual_response = get_contextual_response(input_text, session_id, conversation_history)
        if contextual_response:
            # Store in conversation history
            conversation_contexts[session_id].append({
                'user_input': input_text,
                'response': contextual_response,
                'timestamp': datetime.now().isoformat()
            })
            return contextual_response
        
        # Get relevant context from RAG
        query_embedding = retriever_model.encode([input_text], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, top_k)
        best_idx = int(indices[0][0])
        best_distance = float(distances[0][0])

        # If a threshold is provided, optionally enforce it
        if similarity_threshold is not None and best_distance > similarity_threshold:
            return "I don't have specific information about that. I can help with EADE's programs, admissions, or campus details. What would you like to know?"

        # Get the stored response as context
        context_response = responses[best_idx].strip()
        context_response = context_response.replace("this should not come", "").strip()

        # Create a prompt for the LLM using the context
        system_prompt = """You are a helpful virtual receptionist for EADE Business School. You should:
- Be friendly and professional
- Provide direct, concise answers in 2-3 sentences maximum
- Focus only on the main answer to the user's question
- Avoid unnecessary elaboration or additional information
- Keep responses brief and to the point

Use the following context information to answer the user's question:"""

        user_prompt = f"Context: {context_response}\n\nUser question: {input_text}\n\nProvide a concise response (2-3 sentences max) focusing on the main answer."

        # Generate response using StableLM
        try:
            output = llm(
                f"<|system|>{system_prompt}<|user|>{user_prompt}<|assistant|>",
                max_tokens=128,
                temperature=0.3,
                top_p=0.8,
                stop=["<|user|>", "<|system|>"],
                echo=False
            )
            generated_response = output["choices"][0]["text"].strip()
        except Exception as e:
            print(f"LLM generation error: {e}")
            # Fallback to original response
            generated_response = context_response

        # Add personal touches based on the query type (keep minimal)
        if 'programs' in input_text.lower() and 'offer' in generated_response.lower():
            generated_response += " Would you like details about a specific program?"
        elif 'location' in input_text.lower() or 'address' in input_text.lower():
            generated_response += " Need directions to our campus?"
        elif 'admission' in input_text.lower() or 'apply' in input_text.lower():
            generated_response += " Interested in the application process?"

        # Store in conversation history
        conversation_contexts[session_id].append({
            'user_input': input_text,
            'response': generated_response,
            'timestamp': datetime.now().isoformat()
        })

        # Keep conversation history manageable (last 10 exchanges)
        if len(conversation_contexts[session_id]) > 10:
            conversation_contexts[session_id] = conversation_contexts[session_id][-10:]

        return generated_response
    except Exception as e:
        print(f"Error in chatbot_response: {str(e)}")  # Debug logging
        return "I'm experiencing a technical issue. Please try again or contact EADE Business School directly."
    
# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'eade_business_school_chatbot_2024'  # For session management

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests and return responses"""
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get or create session ID
        session_id = session.get('session_id')
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(conversation_contexts)}"
            session['session_id'] = session_id
        
        # Get response from chatbot with context
        response = chatbot_response(user_message, session_id)
        
        return jsonify({
            'response': response,
            'status': 'success',
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({
            'error': 'I encountered an error processing your request. Please try again or contact EADE directly.',
            'status': 'error'
        }), 500

@app.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    """Reset the conversation context for the current session"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in conversation_contexts:
            del conversation_contexts[session_id]
        
        # Clear session
        session.clear()
        
        return jsonify({
            'message': 'Conversation reset successfully',
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': 'Error resetting conversation. Please try again.',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("Starting EADE Business School Chatbot...")
    print("Access the chatbot at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)