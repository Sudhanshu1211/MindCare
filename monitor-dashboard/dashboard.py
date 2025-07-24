"""
Admin dashboard for monitoring federated learning and risk statistics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sqlite3
from typing import Dict, List, Optional, Tuple

# Page config
st.set_page_config(
    page_title="MindCare - FL Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {font-size:24px !important; font-weight: bold; color: #1f77b4;}
    .sub-header {font-size:20px !important; color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px;}
    .metric-card {padding: 15px; border-radius: 10px; background-color: #f8f9fa; margin: 10px 0;}
    .stProgress > div > div > div > div {background-color: #1f77b4;}
    </style>
""", unsafe_allow_html=True)

# Mock data - In a real app, this would come from your database/APIs
@st.cache_data
def load_mock_data():
    # Generate some mock data for demonstration
    dates = pd.date_range(end=datetime.now(), periods=30).tolist()
    return {
        'active_clients': np.random.randint(1, 10, 30).tolist(),
        'training_rounds': np.random.randint(1, 50, 30).tolist(),
        'accuracy': np.clip(np.random.normal(0.8, 0.1, 30), 0, 1).tolist(),
        'loss': np.clip(np.random.normal(0.3, 0.1, 30), 0, 1).tolist(),
        'dates': dates,
        'latest_round': {
            'round': 29,
            'participants': 8,
            'avg_accuracy': 0.84,
            'avg_loss': 0.27,
            'duration': '2m 15s',
            'completed_at': (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        },
        'client_metrics': [
            {'client_id': f'client_{i}', 'accuracy': np.clip(np.random.normal(0.8, 0.1), 0.6, 0.95), 
             'loss': np.clip(np.random.normal(0.3, 0.1), 0.1, 0.5), 
             'samples': np.random.randint(100, 1000)}
            for i in range(1, 9)
        ]
    }

def display_overview_metrics(data: Dict):
    """Display key metrics in a row"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Clients", f"{data['latest_round']['participants']}", "+2 from last round")
    with col2:
        st.metric("Training Round", f"#{data['latest_round']['round']}", "+1")
    with col3:
        st.metric("Avg. Accuracy", f"{data['latest_round']['avg_accuracy']:.2%}", "+2.5%")
    with col4:
        st.metric("Avg. Loss", f"{data['latest_round']['avg_loss']:.4f}", "-0.012")

def display_training_metrics(data: Dict):
    """Display training metrics charts"""
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Date': data['dates'],
        'Active Clients': data['active_clients'],
        'Training Rounds': data['training_rounds'],
        'Accuracy': data['accuracy'],
        'Loss': data['loss']
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Accuracy Over Time")
        fig = px.line(df, x='Date', y='Accuracy', 
                     title="Model Accuracy Trend",
                     labels={'value': 'Accuracy', 'Date': 'Date'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Model Loss Over Time")
        fig = px.line(df, x='Date', y='Loss', 
                     title="Model Loss Trend",
                     labels={'value': 'Loss', 'Date': 'Date'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def display_client_metrics(data: Dict):
    """Display client-specific metrics"""
    st.markdown("### Client Performance")
    client_df = pd.DataFrame(data['client_metrics'])
    
    # Sort by accuracy
    client_df = client_df.sort_values('accuracy', ascending=False)
    
    # Create a bar chart for client accuracy
    fig = px.bar(client_df, 
                 x='client_id', 
                 y='accuracy',
                 title="Client Model Accuracy",
                 labels={'client_id': 'Client ID', 'accuracy': 'Accuracy'})
    st.plotly_chart(fig, use_container_width=True)

def display_system_health(data: Dict):
    """Display system health metrics"""
    st.markdown("### System Health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Resource Usage")
        # Mock resource usage
        resources = {
            'CPU': 65,
            'Memory': 45,
            'GPU': 30,
            'Network': 15
        }
        
        for name, value in resources.items():
            st.markdown(f"**{name}**")
            st.progress(value / 100)
    
    with col2:
        st.markdown("#### Latest Round Details")
        st.json({
            "Round": data['latest_round']['round'],
            "Participants": data['latest_round']['participants'],
            "Avg. Accuracy": f"{data['latest_round']['avg_accuracy']:.2%}",
            "Avg. Loss": f"{data['latest_round']['avg_loss']:.4f}",
            "Duration": data['latest_round']['duration'],
            "Completed At": data['latest_round']['completed_at']
        })

def main():
    # Title and header
    st.title("MindCare - Federated Learning Monitor")
    st.markdown("---")
    
    # Load data
    data = load_mock_data()
    
    # Display metrics
    display_overview_metrics(data)
    
    # Training metrics section
    st.markdown("## Training Metrics")
    display_training_metrics(data)
    
    # Client metrics section
    st.markdown("## Client Performance")
    display_client_metrics(data)
    
    # System health section
    st.markdown("## System Health")
    display_system_health(data)
    
    # Add a refresh button
    if st.button("🔄 Refresh Data"):
        st.experimental_rerun()
    
    # Add a small footer
    st.markdown("---")
    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
