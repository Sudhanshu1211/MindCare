# MindCare: Privacy-Preserving Mental Health Chatbot

A federated learning-based mental health support system that provides privacy-preserving risk assessment and emotional support through natural conversation.

## 🌟 Features

- **Privacy-First Approach**: End-to-end encryption and federated learning ensure user data never leaves their device
- **Emotional Intelligence**: Advanced sentiment and emotion analysis for personalized responses
- **Clinical Assessment**: Integrated PHQ-9 and GAD-7 questionnaires for depression and anxiety screening
- **Federated Learning**: Collaborative model improvement without sharing raw user data
- **Real-time Monitoring**: Admin dashboard for tracking system performance and user engagement

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sudhanshu1211/MindCare.git
   cd MindCare
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory with:
   ```
   ENCRYPTION_KEY=your_secure_encryption_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL_ID=gemini-1.5-pro
   ```

## 🏗️ Project Structure

```
MindCare/
├── chatbot-ui/           # Streamlit-based chat interface
├── common/               # Shared utilities and models
├── federated-learning-node/  # Flower client for FL
├── fl-server/            # Federated learning server
├── mental-health-api/    # FastAPI backend
├── monitor-dashboard/    # Admin monitoring dashboard
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚦 Running the Application

### 1. Start the Backend API
```bash
cd mental-health-api
uvicorn main:app --reload
```

### 2. Launch the Chatbot UI
```bash
cd chatbot-ui
streamlit run app.py
```

### 3. Start the Federated Learning Server
```bash
cd fl-server
python server.py
```

### 4. Monitor with Dashboard
```bash
cd monitor-dashboard
streamlit run dashboard.py
```

## 🔒 Privacy & Security

- All user data is encrypted at rest and in transit
- Federated learning ensures raw data never leaves user devices
- Model updates are aggregated securely without exposing individual contributions
- Local storage uses AES-256 encryption

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any questions or feedback, please open an issue or contact the maintainers.
