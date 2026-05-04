"""
Streamlit UI for mental health chatbot - Enhanced Interactive Version.
"""

import streamlit as st
import requests
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page Config
st.set_page_config(
    page_title="Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown(
        """
        <style>
        .chat-bubble {
            border-radius: 16px;
            padding: 12px 18px;
            margin-bottom: 10px;
            max-width: 80%;
            word-break: break-word;
            font-size: 1.1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .chat-bubble.user {
            background: #e0f7fa;
            color: #222;
            margin-left: auto;
            margin-right: 0;
        }
        .chat-bubble.bot {
            background: #263238;
            color: #fff;
            margin-right: auto;
            margin-left: 0;
        }
        .sender-label {
            font-weight: bold;
            margin-right: 8px;
        }
        .emotion-badge {
            border-radius: 8px;
            padding: 2px 8px;
            margin-left: 10px;
            font-size: 0.9em;
            color: #fff;
        }
        .emotion-badge.happy { background: #4caf50; }
        .emotion-badge.sad { background: #2196f3; }
        .emotion-badge.angry { background: #f44336; }
        .emotion-badge.neutral { background: #757575; }
        .emotion-badge.surprised { background: #ff9800; }
        .sentiment-badge {
            border-radius: 8px;
            padding: 2px 8px;
            margin-left: 6px;
            font-size: 0.9em;
            color: #fff;
        }
        .sentiment-badge.positive { background: #43a047; }
        .sentiment-badge.negative { background: #e53935; }
        .sentiment-badge.neutral { background: #757575; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_chat_history(chat_history):
    for entry in chat_history:
        sender = entry.get("sender", "bot")
        message = entry.get("message", "")
        emotion = entry.get("emotion", "")
        sentiment = entry.get("sentiment", "")
        sender_class = "user" if sender == "user" else "bot"
        sender_label = "You" if sender == "user" else "Bot"
        emotion_class = emotion.lower() if emotion else "neutral"
        sentiment_class = sentiment.lower() if sentiment else "neutral"
        st.markdown(
            f"""
            <div class="chat-bubble {sender_class}">
                <span class="sender-label">{sender_label}</span>
                <span class="message-text">{message}</span>
                <span class="emotion-badge {emotion_class}">{emotion}</span>
                <span class="sentiment-badge {sentiment_class}">{sentiment}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Initialize session state
inject_custom_css()

# Initialize all session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "quick_message" not in st.session_state:
    st.session_state.quick_message = ""
if "questionnaire_started" not in st.session_state:
    st.session_state.questionnaire_started = False
if "phq9_answers" not in st.session_state:
    st.session_state.phq9_answers = []
if "gad7_answers" not in st.session_state:
    st.session_state.gad7_answers = []

# Define questionnaire questions
phq9_questions = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself",
    "Trouble concentrating on things",
    "Moving or speaking so slowly that others have noticed, or the opposite",
    "Thoughts that you would be better off dead"
]

gad7_questions = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it's hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen"
]

if "phq9_questions" not in st.session_state:
    st.session_state.phq9_questions = phq9_questions
if "gad7_questions" not in st.session_state:
    st.session_state.gad7_questions = gad7_questions

# Set user ID
user_id = st.session_state.get("user_id", "default_user")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "📜 History", "❓ Assessment"])

# ============ TAB 1: CHAT ============
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🚀 Quick Actions")
        if st.button("😊 I'm doing well", use_container_width=True):
            st.session_state.quick_message = "I'm feeling great today!"
        if st.button("😟 I'm stressed", use_container_width=True):
            st.session_state.quick_message = "I've been feeling stressed lately"
        if st.button("😔 I'm sad", use_container_width=True):
            st.session_state.quick_message = "I've been feeling down"
        if st.button("😰 I'm anxious", use_container_width=True):
            st.session_state.quick_message = "I'm feeling anxious and worried"
    
    with col1:
        st.subheader("💬 Conversation")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                for entry in st.session_state.chat_history:
                    # User message
                    st.markdown(f"""
                        <div class="chat-bubble user">
                            <span class="sender-label">You</span>
                            <span class="message-text">{entry.get('user', '')}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Bot message with emotion and sentiment
                    emotion = entry.get('emotion', 'neutral')
                    sentiment = entry.get('sentiment', 'neutral')
                    emotion_class = emotion.lower() if emotion else "neutral"
                    sentiment_class = sentiment.lower() if sentiment else "neutral"
                    
                    st.markdown(f"""
                        <div class="chat-bubble bot">
                            <span class="sender-label">Bot</span>
                            <span class="message-text">{entry.get('bot', '')}</span>
                            <span class="emotion-badge {emotion_class}">{emotion}</span>
                            <span class="sentiment-badge {sentiment_class}">{sentiment}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("💬 Start a conversation to see messages here")
        
        st.divider()
        
        # Input area
        with st.form(key="chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([5, 1])
            with col_input:
                user_msg = st.text_input("Type your message:", label_visibility="collapsed")
            with col_send:
                send_button = st.form_submit_button("📤", use_container_width=True)
            
            if send_button and user_msg.strip():
                with st.spinner("🤖 Bot is thinking..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/ask",
                            json={"message": user_msg, "user_id": user_id},
                            timeout=60
                        )
                        data = resp.json()
                        st.session_state.chat_history.append({
                            "user": user_msg,
                            "bot": data["reply"],
                            "emotion": data["emotion"],
                            "sentiment": data["sentiment"],
                            "timestamp": datetime.now()
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        # Quick message handler
        if st.session_state.quick_message and st.session_state.quick_message.strip():
            with st.spinner("🤖 Bot is thinking..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/ask",
                        json={"message": st.session_state.quick_message, "user_id": user_id},
                        timeout=60
                    )
                    data = resp.json()
                    st.session_state.chat_history.append({
                        "user": st.session_state.quick_message,
                        "bot": data.get("reply", ""),
                        "emotion": data.get("emotion", "neutral"),
                        "sentiment": data.get("sentiment", "neutral"),
                        "timestamp": datetime.now()
                    })
                    st.session_state.quick_message = ""
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ============ TAB 2: ANALYTICS ============
with tab2:
    st.header("📈 Your Mental Health Analytics")
    
    if len(st.session_state.chat_history) == 0:
        st.info("💬 Start chatting to see your analytics!")
    else:
        # Mood History Chart - Time Series
        st.subheader("📊 Mood Progression Over Time")
        if len(st.session_state.chat_history) > 0:
            mood_map = {
                "joy": 5, "happy": 5, "positive": 5,
                "neutral": 3,
                "sad": 1, "sadness": 1, "negative": 1,
                "angry": 2, "anger": 2,
                "anxious": 2, "fear": 2, "scared": 2
            }
            
            timeline_df = pd.DataFrame([
                {
                    "Index": i,
                    "Timestamp": e.get("timestamp", datetime.now()),
                    "Emotion": e.get("emotion", "neutral"),
                    "Sentiment": e.get("sentiment", "neutral"),
                    "Mood_Score": mood_map.get(e.get("emotion", "neutral").lower(), 3)
                }
                for i, e in enumerate(st.session_state.chat_history)
            ])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timeline_df["Timestamp"],
                y=timeline_df["Mood_Score"],
                mode='lines+markers',
                name='Mood Score',
                line=dict(color='#4CAF50', width=3),
                marker=dict(size=8),
                fill='tozeroy'
            ))
            
            fig.update_layout(
                title="Your Mood Progression",
                xaxis_title="Time",
                yaxis_title="Mood Level (1=Sad, 5=Happy)",
                hovermode='x unified',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Emotion and Sentiment Distribution
        st.subheader("📊 Emotion & Sentiment Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            emotions = [e.get("emotion", "neutral") for e in st.session_state.chat_history]
            emotion_counts = pd.Series(emotions).value_counts()
            
            fig = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="😊 Emotion Distribution",
                color_discrete_map={
                    "happy": "#FFD700",
                    "sadness": "#87CEEB",
                    "anger": "#FF6347",
                    "fear": "#9932CC",
                    "neutral": "#CCCCCC",
                    "joy": "#FFD700",
                    "sad": "#87CEEB"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            sentiments = [e.get("sentiment", "neutral") for e in st.session_state.chat_history]
            sentiment_counts = pd.Series(sentiments).value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="💭 Sentiment Distribution",
                color_discrete_map={
                    "positive": "#90EE90",
                    "negative": "#FFB6C6",
                    "neutral": "#FFFACD"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Timeline
        st.subheader("📅 Detailed Conversation Timeline")
        if len(timeline_df) > 0:
            display_df = pd.DataFrame([
                {
                    "Message #": i + 1,
                    "Timestamp": e.get("timestamp", datetime.now()).strftime("%Y-%m-%d %H:%M:%S") if isinstance(e.get("timestamp"), datetime) else str(e.get("timestamp", "N/A")),
                    "Emotion": e.get("emotion", "N/A"),
                    "Sentiment": e.get("sentiment", "N/A")
                }
                for i, e in enumerate(st.session_state.chat_history)
            ])
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Statistics
            st.subheader("📊 Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Conversations", len(st.session_state.chat_history))
            with col2:
                most_common_emotion = emotions[0] if emotions else "N/A"
                st.metric("Most Common Emotion", most_common_emotion)
            with col3:
                avg_mood = sum([mood_map.get(e.lower(), 3) for e in emotions]) / len(emotions) if emotions else 0
                st.metric("Average Mood", f"{avg_mood:.1f}/5")
            with col4:
                most_common_sentiment = sentiments[0] if sentiments else "N/A"
                st.metric("Most Common Sentiment", most_common_sentiment)

# ============ TAB 3: HISTORY ============
with tab3:
    st.header("💾 Chat History")
    
    if len(st.session_state.chat_history) == 0:
        st.info("No conversations yet. Start chatting!")
    else:
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download as JSON"):
                json_str = json.dumps(st.session_state.chat_history, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="chat_history.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📥 Download as CSV"):
                csv_data = pd.DataFrame([
                    {
                        "User Message": e.get("user", ""),
                        "Bot Reply": e.get("bot", ""),
                        "Emotion": e.get("emotion", "N/A"),
                        "Sentiment": e.get("sentiment", "N/A"),
                        "Timestamp": e.get("timestamp", "N/A")
                    }
                    for e in st.session_state.chat_history
                ]).to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="chat_history.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        
        st.divider()
        
        # Display all conversations
        for i, entry in enumerate(st.session_state.chat_history):
            with st.expander(f"Conversation {i+1}: {entry['user'][:50]}..."):
                st.write(f"**You:** {entry['user']}")
                st.write(f"**Bot:** {entry['bot']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"😊 Emotion: **{entry.get('emotion', 'N/A')}**")
                with col2:
                    st.caption(f"💭 Sentiment: **{entry.get('sentiment', 'N/A')}**")

# ============ TAB 4: MENTAL HEALTH ASSESSMENT ============
with tab4:
    st.header("📋 Mental Health Assessment")
    
    if not st.session_state.questionnaire_started:
        st.info("📊 Take the PHQ-9 and GAD-7 questionnaires for a comprehensive mental health assessment")
        if st.button("🚀 Start Questionnaire", use_container_width=True):
            st.session_state.questionnaire_started = True
            st.session_state.phq9_answers = []
            st.session_state.gad7_answers = []
            st.rerun()
    else:
        # PHQ-9 questions
        if len(st.session_state.phq9_answers) < len(st.session_state.phq9_questions):
            idx = len(st.session_state.phq9_answers)
            q = st.session_state.phq9_questions[idx]
            
            progress = idx / len(st.session_state.phq9_questions)
            st.progress(progress)
            st.subheader(f"PHQ-9 - Question {idx+1}/{len(st.session_state.phq9_questions)}")
            st.write(q)
            
            ans = st.radio(
                "How often have you experienced this?",
                ["0 - Not at all", "1 - Several days", "2 - More than half the days", "3 - Nearly every day"],
                key=f"phq9_q{idx}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Previous"):
                    st.session_state.phq9_answers.pop()
                    st.rerun()
            with col2:
                if st.button("➡️ Next"):
                    st.session_state.phq9_answers.append(int(ans[0]))
                    st.rerun()
        
        # GAD-7 questions
        elif len(st.session_state.gad7_answers) < len(st.session_state.gad7_questions):
            idx = len(st.session_state.gad7_answers)
            q = st.session_state.gad7_questions[idx]
            
            progress = (len(st.session_state.phq9_answers) + idx) / (len(st.session_state.phq9_questions) + len(st.session_state.gad7_questions))
            st.progress(progress)
            st.subheader(f"GAD-7 - Question {idx+1}/{len(st.session_state.gad7_questions)}")
            st.write(q)
            
            ans = st.radio(
                "How often have you experienced this?",
                ["0 - Not at all", "1 - Several days", "2 - More than half the days", "3 - Nearly every day"],
                key=f"gad7_q{idx}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Previous"):
                    st.session_state.gad7_answers.pop()
                    st.rerun()
            with col2:
                if st.button("➡️ Finish"):
                    st.session_state.gad7_answers.append(int(ans[0]))
                    st.rerun()
        
        # Show results
        else:
            st.success("✅ Questionnaire Complete!")
            answers = {
                "phq9": st.session_state.phq9_answers,
                "gad7": st.session_state.gad7_answers
            }
            
            try:
                resp = requests.post(
                    f"{API_URL}/score",
                    json={"answers": answers, "user_id": user_id},
                    timeout=10
                )
                data = resp.json()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 PHQ-9 Score", data['phq9_score'], delta="Depression Level")
                with col2:
                    st.metric("📊 GAD-7 Score", data['gad7_score'], delta="Anxiety Level")
                with col3:
                    risk_level = data['risk_level'].capitalize()
                    risk_emoji = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(risk_level, "⚪")
                    st.metric("⚠️ Risk Level", f"{risk_emoji} {risk_level}")
                
                st.divider()
                st.info(f"💡 **Interpretation:**\n- **PHQ-9**: {data['phq9_score']}/27 (Depression screening)\n- **GAD-7**: {data['gad7_score']}/21 (Anxiety screening)")
                
                if st.button("🔄 Start New Assessment"):
                    st.session_state.questionnaire_started = False
                    st.session_state.phq9_answers = []
                    st.session_state.gad7_answers = []
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Could not get score: {e}")

st.divider()
st.markdown("---")
st.markdown("🔐 **Privacy Notice:** Your conversations are encrypted and only stored locally. Data is never shared without consent.")

