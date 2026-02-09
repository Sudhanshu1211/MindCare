"""
Admin dashboard for monitoring federated learning and risk statistics.
"""

import streamlit as st
import json
from datetime import datetime
import pandas as pd

st.set_page_config(
    page_title="MindCare Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {font-size:24px !important; font-weight: bold; color: #1f77b4;}
    .sub-header {font-size:20px !important; color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

import os

# --- Load Model Metrics ---
@st.cache_data
def load_metrics():
    """Loads model performance metrics from a JSON file using a robust path."""
    try:
        # Construct a reliable path to metrics.json relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        metrics_file_path = os.path.join(script_dir, "metrics.json")
        
        with open(metrics_file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: metrics.json not found at the expected location. Please ensure it exists in the 'monitor-dashboard' folder.")
        return None

def load_metrics_history():
    """Loads the history of metrics from metrics_history.json."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    history_file_path = os.path.join(script_dir, "metrics_history.json")
    
    if os.path.exists(history_file_path):
        with open(history_file_path, 'r') as f:
            try:
                history = json.load(f)
                return history
            except (json.JSONDecodeError, TypeError):
                st.warning("Could not read metrics history. File might be empty or corrupt.")
                return []
    return []

def main():
    st.title("🧠 MindCare - Federated Model Monitoring")
    st.markdown("Live dashboard for monitoring the performance of the federated mental health diagnosis model.")

    # --- Accuracy History Chart ---
    st.subheader("📈 Accuracy Over Time")
    metrics_history = load_metrics_history()

    if metrics_history:
        # Convert to pandas DataFrame for easy plotting
        df_history = pd.DataFrame(metrics_history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        df_history = df_history.set_index('timestamp')
        
        # Plot only accuracy
        if 'accuracy' in df_history.columns:
            st.line_chart(df_history[['accuracy']])
        else:
            st.info("No accuracy data found in the history file.")
    else:
        st.info("No metrics history found. Run the client multiple times to see a trend.")

    st.markdown("---_---")

    # --- Current Metrics Scorecard ---
    st.subheader("📊 Current Evaluation Metrics")
    metrics = load_metrics()

    if metrics:
        st.markdown('<p class="sub-header">Model Evaluation Metrics</p>', unsafe_allow_html=True)

        # Display metrics in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        with col2:
            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
        with col3:
            st.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.2f}")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Sensitivity (Recall)", f"{metrics.get('sensitivity', 0):.1%}")
        with col5:
            st.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        with col6:
            st.metric("Specificity", f"{metrics.get('specificity', 0):.1%}")

        st.info("These metrics were calculated on a held-out test dataset to evaluate the model's performance before deployment.")

    # --- Sidebar ---
    st.sidebar.title("About")
    st.sidebar.info(
        "This dashboard provides a static view of the model's offline evaluation metrics. "
        "For live operational monitoring, please refer to the main Federated Learning Dashboard."
    )
    st.sidebar.title("Metric Definitions")
    st.sidebar.markdown("""
    - **Accuracy**: Overall correctness of the model.
    - **Precision**: Of all positive predictions, how many were actually correct.
    - **Sensitivity (Recall)**: Of all actual positive cases, how many did the model correctly identify.
    - **F1 Score**: The harmonic mean of Precision and Recall.
    - **Specificity**: Of all actual negative cases, how many did the model correctly identify.
    - **AUC-ROC**: The model's ability to distinguish between positive and negative classes.
    """)
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Add a small footer
    st.markdown("---")
    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
