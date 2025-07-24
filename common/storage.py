import sqlite3
import os

# Define the path for the database in a user-specific but accessible location
# For simplicity, we'll place it in the project root for now.
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_user_data.db')

def get_db_connection():
    """Creates a database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    print(f"Initializing database at: {DATABASE_PATH}")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create chat_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            encrypted_message TEXT NOT NULL,
            encrypted_reply TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    
    # Create questionnaire_responses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questionnaire_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            encrypted_answers TEXT NOT NULL,
            encrypted_risk_assessment TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def save_chat_message(user_id: str, encrypted_message: str, encrypted_reply: str):
    """Saves an encrypted chat message and reply to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (user_id, encrypted_message, encrypted_reply) VALUES (?, ?, ?)",
        (user_id, encrypted_message, encrypted_reply)
    )
    conn.commit()
    conn.close()
    print(f"Saved chat message for user {user_id[:8]}...")

def save_questionnaire_response(user_id: str, encrypted_answers: str, encrypted_risk: str):
    """Saves an encrypted questionnaire response to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO questionnaire_responses (user_id, encrypted_answers, encrypted_risk_assessment) VALUES (?, ?, ?)",
        (user_id, encrypted_answers, encrypted_risk)
    )
    conn.commit()
    conn.close()
    print(f"Saved questionnaire response for user {user_id[:8]}...")

if __name__ == '__main__':
    # This allows us to run the script directly to initialize the DB
    init_db()
