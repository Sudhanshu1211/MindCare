"""
Streamlit UI for mental health chatbot.
"""

import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠")
st.title("🧠 Federated Mental Health Chatbot")

# Session state for chat and questionnaire
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "questionnaire_started" not in st.session_state:
    st.session_state.questionnaire_started = False
if "phq9_answers" not in st.session_state:
    st.session_state.phq9_answers = []
if "gad7_answers" not in st.session_state:
    st.session_state.gad7_answers = []
if "phq9_questions" not in st.session_state or "gad7_questions" not in st.session_state:
    # Fetch questions from backend
    try:
        qresp = requests.get(f"{API_URL}/get_questions", timeout=5)
        data = qresp.json()
        st.session_state.phq9_questions = data["phq9"]
        st.session_state.gad7_questions = data["gad7"]
    except Exception as e:
        st.error(f"Could not load questions from backend: {e}")
        st.stop()

# Questionnaire logic
if not st.session_state.questionnaire_started:
    st.info("You can start a quick mental health check-in questionnaire or chat with the bot below.")
    if st.button("Start PHQ-9 & GAD-7 Questionnaire"):
        st.session_state.questionnaire_started = True
        st.session_state.phq9_answers = []
        st.session_state.gad7_answers = []

# PHQ-9 questionnaire
if st.session_state.questionnaire_started and len(st.session_state.phq9_answers) < len(st.session_state.phq9_questions):
    idx = len(st.session_state.phq9_answers)
    q = st.session_state.phq9_questions[idx]
    st.subheader(f"PHQ-9 Question {idx+1}/9")
    ans = st.radio(q, ["0 - Not at all", "1 - Several days", "2 - More than half the days", "3 - Nearly every day"])
    if st.button("Next (PHQ-9)"):
        st.session_state.phq9_answers.append(int(ans[0]))
        st.rerun()
# GAD-7 questionnaire
elif st.session_state.questionnaire_started and len(st.session_state.gad7_answers) < len(st.session_state.gad7_questions):
    idx = len(st.session_state.gad7_answers)
    q = st.session_state.gad7_questions[idx]
    st.subheader(f"GAD-7 Question {idx+1}/7")
    ans = st.radio(q, ["0 - Not at all", "1 - Several days", "2 - More than half the days", "3 - Nearly every day"])
    if st.button("Next (GAD-7)"):
        st.session_state.gad7_answers.append(int(ans[0]))
        st.rerun()
# Submit questionnaire
elif st.session_state.questionnaire_started:
    st.success("Questionnaire complete! Calculating your scores...")
    answers = {
        "phq9": st.session_state.phq9_answers,
        "gad7": st.session_state.gad7_answers
    }
    try:
        resp = requests.post(f"{API_URL}/score", json={"answers": answers, "user_id": "user1"}, timeout=10)
        data = resp.json()
        st.write(f"**PHQ-9 Score:** {data['phq9_score']}  ")
        st.write(f"**GAD-7 Score:** {data['gad7_score']}  ")
        st.write(f"**Risk Level:** {data['risk_level'].capitalize()}")
    except Exception as e:
        st.error(f"Could not get score from backend: {e}")
    if st.button("Finish & Return to Chat"):
        st.session_state.questionnaire_started = False
        st.session_state.phq9_answers = []
        st.session_state.gad7_answers = []
        st.rerun()

# Chatbot interface
if not st.session_state.questionnaire_started:
    st.subheader("💬 Chat with the Bot")
    for entry in st.session_state.chat_history:
        st.markdown(f"**You:** {entry['user']}")
        st.markdown(f"**Bot:** {entry['bot']}  ")
        st.caption(f"_Emotion: {entry['emotion']} | Sentiment: {entry['sentiment']}_")
    
    # Use form to enable Enter key submission
    with st.form(key="chat_form", clear_on_submit=True):
        user_msg = st.text_input("Type your message and press Enter or click Send:", key="user_msg_input")
        send_button = st.form_submit_button("Send")
        
        if send_button and user_msg.strip():
            try:
                resp = requests.post(f"{API_URL}/ask", json={"message": user_msg, "user_id": "user1"}, timeout=60)
                data = resp.json()
                st.session_state.chat_history.append({
                    "user": user_msg,
                    "bot": data["reply"],
                    "emotion": data["emotion"],
                    "sentiment": data["sentiment"]
                })
                st.rerun()
            except Exception as e:
                st.error(f"Could not get bot reply: {e}")
