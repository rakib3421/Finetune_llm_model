import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import re

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

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
            return "Hello again! How else can I help you with EADE Business School today?"
    
    # Check if it's a farewell
    if detect_farewell(input_text):
        return get_farewell_response(input_text)
    
    # Check for follow-up questions
    if len(conversation_history) > 0:
        last_response = conversation_history[-1].get('response', '')
        if 'program' in last_response.lower() and any(word in input_text.lower() for word in ['tell me more', 'details', 'more info', 'what about', 'how about']):
            return "I'd be happy to provide more details! Could you specify which program you're interested in? We offer Bachelor programs in Business Administration, Marketing and Sales, Industrial Engineering, and Organizational Psychology, plus various Master's and combined degree programs."
    
    # If the question seems too general or unclear, provide helpful guidance
    if len(input_text.split()) < 3 and not any(word in input_text.lower() for word in ['what', 'how', 'when', 'where', 'why', 'who']):
        return "I'm here to help! Could you please be a bit more specific about what you'd like to know about EADE Business School? I can provide information about our programs, admissions, campus locations, or any other questions you might have."
    
    return None

def get_greeting_response(input_text):
    """Get appropriate greeting response"""
    current_hour = datetime.now().hour
    
    if 'morning' in input_text.lower() or (5 <= current_hour < 12):
        return "Good morning! What a wonderful day to learn about EADE Business School. I'm your virtual receptionist, ready to help you with information about our programs, application process, facilities, or any other questions. How can I assist you this morning?"
    elif 'afternoon' in input_text.lower() or (12 <= current_hour < 17):
        return "Good afternoon! I hope you're having a great day. Welcome to EADE Business School's information desk. I'm here to help you with questions about our business programs, admissions, campus life, or anything else you'd like to know. What brings you here today?"
    elif 'evening' in input_text.lower() or (17 <= current_hour < 22):
        return "Good evening! Thank you for visiting EADE Business School. Even though it's evening, I'm here and ready to help you with any information you need about our programs, admissions process, or facilities. What would you like to know?"
    else:
        return "Hello! Welcome to EADE Business School. I'm your virtual receptionist, and I'm here to assist you with any questions about our programs, admissions, campus facilities, or anything else you'd like to know. How may I help you today?"

def get_farewell_response(input_text):
    """Get appropriate farewell response"""
    if 'thank' in input_text.lower():
        return "You're very welcome! I'm so glad I could help you today. If you have any more questions about EADE Business School, our programs, or anything else, please don't hesitate to ask. I'm always here to assist prospective and current students. Have a wonderful day!"
    else:
        return "Goodbye! It was wonderful talking with you today. Thank you for your interest in EADE Business School. I hope to see you again soon, whether as a prospective student or when you need more information. Have a fantastic day!"

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
        
        # Use RAG for specific questions
        query_embedding = retriever_model.encode([input_text], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, top_k)
        best_idx = int(indices[0][0])
        best_distance = float(distances[0][0])

        # If a threshold is provided, optionally enforce it
        if similarity_threshold is not None and best_distance > similarity_threshold:
            return "I'm sorry, I don't have specific information about that in my knowledge base. However, I'd be happy to help you with questions about EADE Business School's programs, admissions, campus locations, or connect you with someone who can provide more detailed assistance. What specific aspect of EADE would you like to know about?"

        # Return the stored response exactly (trim whitespace). If long, return first sentence.
        resp = responses[best_idx].strip()
        # Remove unwanted phrase if present
        resp = resp.replace("this should not come", "").strip()
        
        # Add a personal touch as a receptionist
        if 'programs' in input_text.lower() and 'offer' in resp.lower():
            resp += " Would you like me to provide more details about any specific program that interests you?"
        elif 'location' in input_text.lower() or 'address' in input_text.lower():
            resp += " Would you like directions or information about visiting our campus?"
        elif 'admission' in input_text.lower() or 'apply' in input_text.lower():
            resp += " I can provide more details about the application process if you're interested!"
        
        # Return only the first sentence to keep answers concise, but more conversational
        if '.' in resp:
            sentences = resp.split('.')
            if len(sentences) > 1 and len(sentences[0].strip()) > 0:
                response = sentences[0].strip() + '.'
                # # Add follow-up if appropriate
                # if not any(word in response.lower() for word in ['would you like', 'can provide', 'i can']):
                #     response += " Is there anything specific you'd like to know more about?"
                # resp = response
        
        # Store in conversation history
        conversation_contexts[session_id].append({
            'user_input': input_text,
            'response': resp,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep conversation history manageable (last 10 exchanges)
        if len(conversation_contexts[session_id]) > 10:
            conversation_contexts[session_id] = conversation_contexts[session_id][-10:]
        
        return resp
    except Exception as e:
        print(f"Error in chatbot_response: {str(e)}")  # Debug logging
        return f"I apologize, but I'm experiencing a technical issue right now. Please try again in a moment, or feel free to contact EADE Business School directly for immediate assistance."
    
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
            'error': f'I apologize, but I encountered an error while processing your request. Please try again or contact EADE Business School directly for assistance. Error: {str(e)}',
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
            'error': f'Error resetting conversation: {str(e)}',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("Starting EADE Business School Chatbot...")
    print("Access the chatbot at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)