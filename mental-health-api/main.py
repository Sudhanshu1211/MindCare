"""
FastAPI backend for NLP, diagnosis, and encrypted data handling.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Any
import google.generativeai as genai
from common.models import Gemini

# Load environment variables from the project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "test-key")
MODEL_PATH = os.getenv("MODEL_PATH", "distilbert-base-uncased")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID = "models/gemini-2.0-flash"
if GEMINI_API_KEY:
    gemini_agent = Gemini(api_key=GEMINI_API_KEY, id=GEMINI_MODEL_ID, temprature=0.2)

# Import encryption utilities
from common.crypto_utils import encrypt_data, decrypt_data

from common.models import get_sentiment_pipeline, get_emotion_pipeline
from common.storage import init_db, save_chat_message, save_questionnaire_response

app = FastAPI(title="Mental Health Diagnosis API")

@app.on_event("startup")
async def on_startup():
    """Ensure the database is initialized when the application starts."""
    init_db()

# Load sentiment and emotion pipelines from common.models
print("\n[DEBUG] Initializing sentiment pipeline...")
try:
    sentiment_pipeline = get_sentiment_pipeline()
    HAS_SENTIMENT = True
    print(f"[DEBUG] Successfully loaded sentiment pipeline: {sentiment_pipeline}")
except Exception as e:
    sentiment_pipeline = None
    HAS_SENTIMENT = False
    print(f"[DEBUG] Failed to load sentiment pipeline: {str(e)}")
    import traceback
    print(f"[DEBUG] Error details: {traceback.format_exc()}")

try:
    emotion_pipeline = get_emotion_pipeline()
    HAS_EMOTION = True
except Exception:
    emotion_pipeline = None
    HAS_EMOTION = False

# Encryption helper functions
def encrypt_user_data(data: str) -> str:
    """Encrypt sensitive user data"""
    try:
        encrypted_bytes = encrypt_data(data.encode('utf-8'), ENCRYPTION_KEY)
        return encrypted_bytes.hex()  # Convert to hex string for storage
    except Exception as e:
        print(f"Encryption error: {e}")
        return data  # Return original data if encryption fails

def decrypt_user_data(encrypted_hex: str) -> str:
    """Decrypt sensitive user data"""
    try:
        encrypted_bytes = bytes.fromhex(encrypted_hex)
        decrypted_bytes = decrypt_data(encrypted_bytes, ENCRYPTION_KEY)
        return decrypted_bytes.decode('utf-8')
    except Exception as e:
        print(f"Decryption error: {e}")
        return encrypted_hex  # Return original data if decryption fails

# PHQ-9 and GAD-7 questions (static for now)
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself - or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead, or thoughts of hurting yourself in some way"
]

GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen"
]

class AskRequest(BaseModel):
    message: str
    user_id: str

class AskResponse(BaseModel):
    reply: str
    emotion: str
    sentiment: str
    encrypted_reply: str = None  # Optional encrypted version
    encrypted_message: str = None  # Optional encrypted user message

class ScoreRequest(BaseModel):
    answers: Dict[str, List[int]]  # {"phq9": [...], "gad7": [...]}
    user_id: str

class ScoreResponse(BaseModel):
    phq9_score: int
    gad7_score: int
    risk_level: str

@app.get("/get_questions")
def get_questions():
    """Return PHQ-9 and GAD-7 question sets."""
    return {"phq9": PHQ9_QUESTIONS, "gad7": GAD7_QUESTIONS}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Accepts user response and returns chatbot reply with emotion/sentiment.
    Uses Gemini agent for reply if API key is set, else falls back to template. Sentiment and emotion analysis are performed on the reply.
    User data is encrypted for privacy.
    """
    # Encrypt user message for secure logging/storage
    encrypted_message = encrypt_user_data(request.message)
    print(f"[Security] User message encrypted: {encrypted_message[:20]}...")
    
    # Log encrypted user ID for audit trail
    encrypted_user_id = encrypt_user_data(request.user_id)
    print(f"[Security] User ID encrypted: {encrypted_user_id[:20]}...")
    
    reply = None
    # Only use Gemini agent for replies
    if GEMINI_API_KEY:
        try:
            gemini_prompt = (
                "You are a compassionate mental health support chatbot. "
                "Respond helpfully and empathetically.\nUser: " + request.message
            )
            print("[Gemini] Generating reply...")
            gemini_response = gemini_agent.generate(gemini_prompt)
            reply = gemini_response.text.strip()
            print(f"[Gemini reply]: {reply}")
        except Exception as e:
            reply = f"[Gemini error: {e}]"
    if not reply:
        reply = f"Thank you for sharing. I understand you feel neutral."
    
    # Encrypt the reply for secure storage/transmission
    encrypted_reply = encrypt_user_data(reply)
    print(f"[Security] Reply encrypted: {encrypted_reply[:20]}...")

    # Sentiment analysis on the USER'S message (not the reply)
    sentiment = "neutral"  # Default value
    if HAS_SENTIMENT and sentiment_pipeline:
        try:
            # Use the original user message for sentiment analysis
            user_message = request.message
            print(f"\n[DEBUG] Analyzing sentiment for user message: {user_message[:100]}...")
            
            # Get sentiment scores for all classes
            sentiment_result = sentiment_pipeline(user_message, truncation=True, max_length=512, top_k=None)
            print(f"[DEBUG] Raw sentiment result: {sentiment_result}")
            
            if sentiment_result and isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                # The model returns a list of dictionaries with 'label' and 'score'
                scores = {}
                # sentiment_result is already a list of dictionaries, no need for [0]
                for item in sentiment_result:
                    if isinstance(item, dict):
                        label = str(item.get('label', '')).lower()
                        score = float(item.get('score', 0.0))
                        scores[label] = score
                        print(f"[DEBUG] {label}: {score:.4f}")
                
                # Get the label with highest score
                if scores:
                    sentiment = max(scores.items(), key=lambda x: x[1])[0]
                    confidence = scores[sentiment]
                    print(f"[Sentiment Analysis] Final sentiment: {sentiment} (confidence: {confidence:.4f})")
                else:
                    print("[DEBUG] No valid sentiment scores found")
                
        except Exception as e:
            import traceback
            print(f"[Error in sentiment analysis]: {str(e)}")
            print(f"[Error details]: {traceback.format_exc()}")
            sentiment = "error"
    # Emotion analysis on the user's message (if available)
    emotion = "neutral"  # Default value
    if HAS_EMOTION and emotion_pipeline:
        try:
            print(f"\n[DEBUG] Analyzing emotion for user message: {user_message[:100]}...")
            emotion_result = emotion_pipeline(user_message, truncation=True, max_length=512, top_k=1)
            print(f"[DEBUG] Raw emotion result: {emotion_result}")
            
            if emotion_result and isinstance(emotion_result, list) and len(emotion_result) > 0:
                # Get the first (and most likely) emotion
                first_emotion = emotion_result[0]
                if isinstance(first_emotion, dict):
                    emotion = str(first_emotion.get('label', 'neutral')).lower()
                    score = float(first_emotion.get('score', 0.0))
                    print(f"[Emotion Analysis] Detected emotion: {emotion} (confidence: {score:.4f})")
        except Exception as e:
            import traceback
            print(f"[Error in emotion analysis]: {str(e)}")
            print(f"[Error details]: {traceback.format_exc()}")
    # Save the encrypted conversation to the database
    try:
        save_chat_message(
            user_id=encrypted_user_id, 
            encrypted_message=encrypted_message, 
            encrypted_reply=encrypted_reply
        )
    except Exception as e:
        print(f"[Database Error] Failed to save chat message: {e}")

    # Return response with both plain and encrypted data
    return AskResponse(
        reply=reply,
        emotion=emotion,
        sentiment=sentiment,
        encrypted_reply=encrypted_reply,
        encrypted_message=encrypted_message
    )

@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    """
    Returns depression/anxiety risk scores based on answers.
    Questionnaire data is encrypted for privacy.
    """
    # Encrypt user ID for secure logging
    encrypted_user_id = encrypt_user_data(request.user_id)
    print(f"[Security] Questionnaire user ID encrypted: {encrypted_user_id[:20]}...")
    
    # Encrypt questionnaire answers for secure storage
    encrypted_answers = encrypt_user_data(str(request.answers))
    print(f"[Security] Questionnaire answers encrypted: {encrypted_answers[:20]}...")
    
    phq9_score = sum(request.answers.get("phq9", []))
    gad7_score = sum(request.answers.get("gad7", []))
    
    # Simple risk logic (to be improved)
    risk_level = "low"
    if phq9_score >= 15 or gad7_score >= 15:
        risk_level = "high"
    elif phq9_score >= 10 or gad7_score >= 10:
        risk_level = "moderate"
    
    # Encrypt risk assessment for secure storage
    encrypted_risk = encrypt_user_data(f"PHQ9:{phq9_score}, GAD7:{gad7_score}, Risk:{risk_level}")
    print(f"[Security] Risk assessment encrypted: {encrypted_risk[:20]}...")

    # Save the encrypted questionnaire response to the database
    try:
        save_questionnaire_response(
            user_id=encrypted_user_id,
            encrypted_answers=encrypted_answers,
            encrypted_risk=encrypted_risk
        )
    except Exception as e:
        print(f"[Database Error] Failed to save questionnaire response: {e}")
    
    return ScoreResponse(
        phq9_score=phq9_score,
        gad7_score=gad7_score,
        risk_level=risk_level
    )
