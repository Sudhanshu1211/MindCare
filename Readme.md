# MentalEase

## How to Run

1. **Clone the repository**
   ```sh
   git clone <repository-url>
   ```
2. **Create a virtual environment**
   ```sh
   python -m venv venv
   ```
3. **Activate the virtual environment**
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```
4. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
5. **Create a `.env` file** and store your Google API key
   ```env
   GOOGLE_API_KEY="your_api_key"
   ```
6. **Run the application**
   ```sh
   streamlit run app.py
   ```

## Requirements

- Python 3.x
- pip

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.