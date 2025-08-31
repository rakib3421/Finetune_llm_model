import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import re
import os
import threading
import time
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from llama_cpp import Llama
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

# Load conversations for learning (create if doesn't exist)
conversations_file = 'conversations.json'
if os.path.exists(conversations_file):
    with open(conversations_file, 'r') as f:
        conversation_data = json.load(f)
else:
    conversation_data = []

# Enhanced NLP Processing Class
class AdvancedNLPProcessor:
    def __init__(self):
        # Intent classification patterns
        self.intent_patterns = {
            'admission': [
                r'\b(admission|apply|application|enroll|registration|requirements|eligibility)\b',
                r'\bhow to (apply|join|enroll|register)\b',
                r'\bapplication (process|deadline|requirements)\b'
            ],
            'programs': [
                r'\b(program|course|degree|major|curriculum|subjects|modules)\b',
                r'\bwhat (programs|courses|degrees) (do you|does)\b',
                r'\b(undergraduate|postgraduate|masters|bachelors)\b'
            ],
            'fees': [
                r'\b(fee|fees|cost|price|tuition|payment|scholarship|financial)\b',
                r'\bhow much (does it cost|is the fee|to pay)\b',
                r'\b(scholarship|financial aid|payment plan)\b'
            ],
            'location': [
                r'\b(location|address|where|campus|building|directions|map)\b',
                r'\bwhere is (the|your|campus|school|university)\b',
                r'\bhow to (get|reach|find)\b.*\b(campus|school)\b'
            ],
            'contact': [
                r'\b(contact|phone|email|call|reach|speak|talk)\b',
                r'\bhow to (contact|reach|call)\b',
                r'\b(phone number|email address|contact details)\b'
            ],
            'facilities': [
                r'\b(facilities|library|lab|laboratory|cafeteria|hostel|accommodation)\b',
                r'\bwhat (facilities|services) (do you|does)\b',
                r'\b(wifi|internet|parking|transport)\b'
            ]
        }
        
        # Context keywords for better understanding
        self.context_keywords = {
            'school_context': ['eade', 'business school', 'university', 'college', 'institution'],
            'academic_context': ['study', 'learn', 'education', 'academic', 'course', 'program'],
            'inquiry_context': ['information', 'details', 'know', 'tell me', 'explain', 'help']
        }
    
    def extract_intent(self, text):
        """Extract intent from user input"""
        text_lower = text.lower()
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        return 'general'
    
    def extract_entities(self, text):
        """Extract entities from text using regex patterns"""
        entities = {}
        text_lower = text.lower()
        
        # Program entities
        programs = re.findall(r'\b(business|marketing|engineering|psychology|computer science|mba|bba)\b', text_lower)
        if programs:
            entities['programs'] = list(set(programs))
        
        # Time entities
        time_refs = re.findall(r'\b(today|tomorrow|next week|this month|next year|2024|2025)\b', text_lower)
        if time_refs:
            entities['time'] = list(set(time_refs))
        
        return entities
    
    def analyze_similarity(self, text1, text2):
        """Analyze similarity between two texts"""
        vectorizer = TfidfVectorizer().fit([text1, text2])
        vectors = vectorizer.transform([text1, text2])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

# Initialize NLP processor
nlp_processor = AdvancedNLPProcessor()

# Learning and Pattern Recognition
class ConversationLearner:
    def __init__(self, data_file='data.json', conversations_file='conversations.json'):
        self.data_file = data_file
        self.conversations_file = conversations_file
        self.learning_threshold = 2  # Minimum frequency for pattern learning
        self.similarity_threshold = 0.7
        
        # Start background learning thread
        self.learning_thread = threading.Thread(target=self.continuous_learning_loop, daemon=True)
        self.learning_thread.start()
    
    def continuous_learning_loop(self):
        """Background thread for continuous learning"""
        while True:
            try:
                time.sleep(300)  # Learn every 5 minutes
                self.learn_from_conversations()
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
    
    def learn_from_conversations(self):
        """Learn new patterns from conversations"""
        try:
            global conversation_data, data, prompts, responses
            
            if len(conversation_data) < 5:  # Need minimum conversations
                return
            
            # Analyze conversation patterns
            patterns = self.find_conversation_patterns()
            new_patterns_added = 0
            
            for pattern in patterns:
                if not self.is_pattern_in_knowledge_base(pattern['user_input']):
                    # Add to knowledge base
                    new_entry = {
                        "prompt": pattern['user_input'],
                        "response": pattern['bot_response'],
                        "learned_from_frequency": pattern['frequency'],
                        "learned_date": datetime.now().isoformat()
                    }
                    data.append(new_entry)
                    new_patterns_added += 1
            
            if new_patterns_added > 0:
                # Update global variables
                prompts = [item['prompt'] for item in data]
                responses = [item['response'] for item in data]
                
                # Update embeddings
                update_knowledge_embeddings()
                
                # Save updated knowledge base
                self.save_knowledge_base()
                logger.info(f"Learned {new_patterns_added} new patterns")
                
        except Exception as e:
            logger.error(f"Error in learning: {e}")
    
    def find_conversation_patterns(self):
        """Find patterns in conversation data"""
        patterns = defaultdict(list)
        
        for conv in conversation_data:
            user_input = conv.get('user_input', '').strip().lower()
            bot_response = conv.get('bot_response', '').strip()
            
            if len(user_input) > 3 and len(bot_response) > 10:
                patterns[user_input].append(bot_response)
        
        # Find frequent patterns
        frequent_patterns = []
        for user_input, responses_list in patterns.items():
            if len(responses_list) >= self.learning_threshold:
                # Use most common response
                most_common_response = max(set(responses_list), key=responses_list.count)
                frequent_patterns.append({
                    'user_input': user_input,
                    'bot_response': most_common_response,
                    'frequency': len(responses_list)
                })
        
        return frequent_patterns
    
    def is_pattern_in_knowledge_base(self, user_input):
        """Check if pattern exists in knowledge base"""
        user_input_lower = user_input.lower().strip()
        for item in data:
            if item['prompt'].lower().strip() == user_input_lower:
                return True
        return False
    
    def save_knowledge_base(self):
        """Save updated knowledge base"""
        try:
            # Create backup
            backup_file = f"{self.data_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save updated version
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")

# Initialize learner
conversation_learner = ConversationLearner()

def save_conversation(session_id, user_input, bot_response, intent=None, entities=None, confidence=None):
    """Save conversation to JSON file"""
    global conversation_data
    
    conversation_entry = {
        'session_id': session_id,
        'user_input': user_input,
        'bot_response': bot_response,
        'intent': intent,
        'entities': entities,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    }
    
    conversation_data.append(conversation_entry)
    
    # Save to file (keep last 1000 conversations)
    if len(conversation_data) > 1000:
        conversation_data = conversation_data[-1000:]
    
    try:
        with open(conversations_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")

def search_conversation_history(query, limit=3):
    """Search similar conversations from history"""
    if not conversation_data:
        return []
    
    query_lower = query.lower()
    similar_conversations = []
    
    for conv in conversation_data:
        user_input = conv.get('user_input', '').lower()
        similarity = nlp_processor.analyze_similarity(query_lower, user_input)
        
        if similarity > 0.3:  # Similarity threshold
            similar_conversations.append({
                'conversation': conv,
                'similarity': similarity
            })
    
    # Sort by similarity and return top results
    similar_conversations.sort(key=lambda x: x['similarity'], reverse=True)
    return similar_conversations[:limit]

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

# Function to update embeddings
def update_knowledge_embeddings():
    """Update embeddings and FAISS index when knowledge base changes"""
    global prompt_embeddings, faiss_index, prompts, responses
    
    prompts = [item['prompt'] for item in data]
    responses = [item['response'] for item in data]
    
    if not prompts:
        return
        
    # Embed prompts
    prompt_embeddings = retriever_model.encode(prompts, convert_to_numpy=True)
    
    # Create FAISS index
    dimension = prompt_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(prompt_embeddings)
    
    logger.info(f"Updated embeddings for {len(prompts)} prompts")

# Initial embedding creation
update_knowledge_embeddings()

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
    """Enhanced chatbot response with learning and advanced NLP"""
    try:
        # Get or create conversation history for this session
        if session_id not in conversation_contexts:
            conversation_contexts[session_id] = []
        
        conversation_history = conversation_contexts[session_id]
        
        # Advanced NLP processing
        intent = nlp_processor.extract_intent(input_text)
        entities = nlp_processor.extract_entities(input_text)
        
        # Check for contextual responses first
        contextual_response = get_contextual_response(input_text, session_id, conversation_history)
        if contextual_response:
            # Save conversation for learning
            save_conversation(session_id, input_text, contextual_response, intent, entities, 0.9)
            
            # Store in conversation history
            conversation_contexts[session_id].append({
                'user_input': input_text,
                'response': contextual_response,
                'intent': intent,
                'entities': entities,
                'timestamp': datetime.now().isoformat()
            })
            return contextual_response
        
        # Search conversation history for similar questions
        similar_conversations = search_conversation_history(input_text)
        
        # Get relevant context from RAG
        query_embedding = retriever_model.encode([input_text], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, top_k)
        best_idx = int(indices[0][0])
        best_distance = float(distances[0][0])
        confidence = max(0.0, 1.0 - (best_distance / 2.0))

        # If a threshold is provided, optionally enforce it
        if similarity_threshold is not None and best_distance > similarity_threshold:
            response = "I don't have specific information about that. I can help with EADE's programs, admissions, or campus details. What would you like to know?"
            save_conversation(session_id, input_text, response, intent, entities, 0.3)
            return response

        # Get the stored response as context
        context_response = responses[best_idx].strip()
        context_response = context_response.replace("this should not come", "").strip()

        # Enhance response using conversation history
        if similar_conversations:
            most_similar = similar_conversations[0]
            if most_similar['similarity'] > 0.7:
                # Use similar conversation response as additional context
                context_response += f" Based on previous conversations: {most_similar['conversation']['bot_response']}"

        # Create enhanced prompt for the LLM
        system_prompt = f"""You are a helpful virtual receptionist for EADE Business School. 

User Intent: {intent}
Detected Information: {entities}
Confidence: {confidence:.2f}

Guidelines:
- Be friendly and professional
- Provide direct, concise answers in 2-3 sentences maximum
- Focus on the specific intent: {intent}
- Use the context information to answer accurately
- If asking about {intent}, prioritize that information

Use the following context information to answer the user's question:"""

        user_prompt = f"Context: {context_response}\n\nUser question: {input_text}\n\nProvide a helpful response focusing on {intent}:"

        # Generate response using StableLM
        try:
            output = llm(
                f"<|system|>{system_prompt}<|user|>{user_prompt}<|assistant|>",
                max_tokens=150,
                temperature=0.3,
                top_p=0.8,
                stop=["<|user|>", "<|system|>"],
                echo=False
            )
            generated_response = output["choices"][0]["text"].strip()
        except Exception as e:
            print(f"LLM generation error: {e}")
            generated_response = context_response

        # Add intent-specific enhancements
        if intent == 'programs' and 'offer' in generated_response.lower():
            generated_response += " Would you like details about a specific program?"
        elif intent == 'location':
            generated_response += " Need directions to our campus?"
        elif intent == 'admission':
            generated_response += " Interested in the application process?"
        elif intent == 'fees':
            generated_response += " Want to know about scholarship opportunities?"

        # Save conversation for learning
        save_conversation(session_id, input_text, generated_response, intent, entities, confidence)

        # Store in conversation history
        conversation_contexts[session_id].append({
            'user_input': input_text,
            'response': generated_response,
            'intent': intent,
            'entities': entities,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })

        # Keep conversation history manageable (last 15 exchanges)
        if len(conversation_contexts[session_id]) > 15:
            conversation_contexts[session_id] = conversation_contexts[session_id][-15:]

        return generated_response
    except Exception as e:
        print(f"Error in chatbot_response: {str(e)}")
        error_response = "I'm experiencing a technical issue. Please try again or contact EADE Business School directly."
        save_conversation(session_id, input_text, error_response, 'error', {}, 0.0)
        return error_response
    
# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'eade_business_school_chatbot_2024'  # For session management

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests and return responses with enhanced analytics"""
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get or create session ID
        session_id = session.get('session_id')
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(conversation_contexts)}"
            session['session_id'] = session_id
        
        # Get response from enhanced chatbot
        response = chatbot_response(user_message, session_id)
        
        # Get intent and confidence from last conversation entry
        last_conv = conversation_contexts.get(session_id, [])
        intent = 'general'
        confidence = 0.5
        entities = {}
        
        if last_conv:
            last_entry = last_conv[-1]
            intent = last_entry.get('intent', 'general')
            confidence = last_entry.get('confidence', 0.5)
            entities = last_entry.get('entities', {})
        
        return jsonify({
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'status': 'success',
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({
            'error': 'I encountered an error processing your request. Please try again or contact EADE directly.',
            'status': 'error'
        }), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle user feedback for learning improvement"""
    try:
        session_id = request.json.get('session_id')
        rating = request.json.get('rating')  # 1 for positive, 0 for negative
        comment = request.json.get('comment', '')
        
        if session_id and rating is not None:
            # Store feedback in conversation data
            feedback_entry = {
                'session_id': session_id,
                'rating': rating,
                'comment': comment,
                'timestamp': datetime.now().isoformat(),
                'type': 'feedback'
            }
            conversation_data.append(feedback_entry)
            
            # Save updated conversation data
            try:
                with open(conversations_file, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error saving feedback: {e}")
            
            return jsonify({'status': 'success', 'message': 'Thank you for your feedback!'})
        else:
            return jsonify({'error': 'Missing required feedback data'}), 400
            
    except Exception as e:
        logger.error(f"Error handling feedback: {e}")
        return jsonify({'error': 'Error storing feedback'}), 500

@app.route('/learning_stats', methods=['GET'])
def learning_stats():
    """Get learning and conversation statistics"""
    try:
        # Count conversations
        total_conversations = len([c for c in conversation_data if c.get('type') != 'feedback'])
        
        # Count feedback
        positive_feedback = len([c for c in conversation_data if c.get('type') == 'feedback' and c.get('rating') == 1])
        negative_feedback = len([c for c in conversation_data if c.get('type') == 'feedback' and c.get('rating') == 0])
        
        # Count learned patterns (entries with learned_date)
        learned_patterns = len([item for item in data if 'learned_date' in item])
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        recent_conversations = len([
            c for c in conversation_data 
            if c.get('type') != 'feedback' and 
            datetime.fromisoformat(c.get('timestamp', '1970-01-01')) >= recent_cutoff
        ])
        
        # Intent distribution
        intent_counts = defaultdict(int)
        for conv in conversation_data:
            if conv.get('type') != 'feedback' and 'intent' in conv:
                intent_counts[conv['intent']] += 1
        
        return jsonify({
            'total_conversations': total_conversations,
            'learned_patterns': learned_patterns,
            'knowledge_base_size': len(data),
            'recent_conversations': recent_conversations,
            'positive_feedback': positive_feedback,
            'negative_feedback': negative_feedback,
            'intent_distribution': dict(intent_counts),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error getting learning stats: {e}")
        return jsonify({'error': 'Error retrieving statistics'}), 500

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