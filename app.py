import os
import streamlit as st
import json
from dotenv import load_dotenv
import google.generativeai as gen_ai
from Agents.agent import Agents
import re

# Load environment variables
load_dotenv()

gen_ai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Agents
agents = Agents()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Depression Detection Chatbot",
    page_icon="🧠",
    layout="centered",
)

def initialize_session_state():
    state_defaults = {
        "chat_session": agents.model.start_chat(history=[]),
        "chat_stage": 0,
        "user_data": {},
        "story_questions": [],
        "situation_question":[],
        "evaluation_data": {},
        "user_response": {},
        "question_index1": 0,
        "question_index2": 0,
    }
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

def reset_and_rerun():
    st.session_state.clear()
    st.rerun()

st.title("Depression Detection Chatbot")
st.write("Let's assess your mental well-being. Please answer a few questions.")
if st.button(label="Restart Process"):
    reset_and_rerun()

# Display Chat History
for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role_for_streamlit(message["role"])):
        st.markdown(message["parts"][0]["text"])

# Define Static Questions
questions = [
    ("Full Name", "What is your full name?"),
    ("Age", "How old are you?"),
    ("Sleep Patterns", "How would you describe your sleep patterns? (e.g., insomnia, excessive sleep)")
]

def ask_questions():
    if st.session_state.chat_stage < len(questions):
        key, question = questions[st.session_state.chat_stage]

        with st.chat_message("assistant"):
            st.markdown(question)

        user_input = st.chat_input("Your response")
        if user_input:
            st.session_state.user_data[key] = user_input
            st.session_state.chat_session.history.extend([
                {"role": "model", "parts": [{"text": question}]},
                {"role": "user", "parts": [{"text": user_input}]},
            ])
            st.session_state.chat_stage += 1
            st.rerun()
    else:
        handle_dynamic_questions()

def handle_dynamic_questions():
    if not st.session_state.situation_question:
        try:
            generated_questions = agents.situation_question_generation_agent(st.session_state.user_data)
            cleaned_string = re.sub(r"```json|```", "", generated_questions).strip()
            st.session_state.situation_question = json.loads(cleaned_string)
        except json.JSONDecodeError:
            st.error("Failed to generate dynamic questions.")
            return
    
    question_keys = list(st.session_state.situation_question.keys())
    question_index1 = st.session_state.question_index1
    
    if question_index1 < len(question_keys):
        key = question_keys[question_index1]
        question = st.session_state.situation_question[key]

        with st.chat_message("assistant"):
            st.markdown(question)

        user_input = st.chat_input("Your response")
        if user_input:
            st.session_state.user_response[question] = user_input
            st.session_state.chat_session.history.extend([
                {"role": "model", "parts": [{"text": question}]},
                {"role": "user", "parts": [{"text": user_input}]},
            ])
            st.session_state.evaluation_data[question] = user_input
            st.session_state.question_index1 += 1
            st.rerun()
    else:
        storytellingquestion()


def storytellingquestion():
    if not st.session_state.story_questions:
        try:
            generated_questions = agents.generate_storytelling_questions_agent(st.session_state.user_data)
            cleaned_string = re.sub(r"```json|```", "", generated_questions).strip()
            st.session_state.story_questions = json.loads(cleaned_string)
        except json.JSONDecodeError:
            st.error("Failed to generate dynamic questions.")
            return
    
    question_keys = list(st.session_state.story_questions.keys())
    question_index2 = st.session_state.question_index2
    
    if question_index2 < len(question_keys):
        key = question_keys[question_index2]
        question = st.session_state.story_questions[key]

        with st.chat_message("assistant"):
            st.markdown(question)

        user_input = st.chat_input("Your response")
        if user_input:
            st.session_state.user_response[question] = user_input
            st.session_state.chat_session.history.extend([
                {"role": "model", "parts": [{"text": question}]},
                {"role": "user", "parts": [{"text": user_input}]},
            ])
            st.session_state.evaluation_data[question] = user_input
            st.session_state.question_index2 += 1
            st.rerun()
    else:
        finalize_evaluation()

def finalize_evaluation():
    try:
        score = agents.evaluate_user(st.session_state.evaluation_data)
        cleaned_score = re.sub(r"```json|```", "", score).strip()
        print("Eval data", st.session_state.evaluation_data)
        st.markdown("Thank you for your time!")
        st.json(cleaned_score)
        print("Candidate Evaluation Score:", cleaned_score)
    except Exception as e:
        st.error("An error occurred during evaluation. Please try again.")
        print("Evaluation Error:", e)


ask_questions()
