import flwr as fl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
from sklearn.model_selection import train_test_split

# --- Add project root to sys.path to allow imports from common --- #
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from common.storage import get_db_connection
from common.crypto_utils import decrypt_data
from dotenv import load_dotenv

# --- Configuration --- #
load_dotenv(os.path.join(project_root, '.env'))
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SAVED_MODEL_PATH = os.path.abspath(os.path.join(project_root, "saved_models", "federated_model.pt"))

# --- 1. Load and Decrypt Data --- #
def load_data():
    """Loads, decrypts, and splits data into training and validation sets."""
    print("Loading and decrypting local data...")
    conn = get_db_connection()
    rows = conn.execute("SELECT encrypted_message, encrypted_reply FROM chat_history").fetchall()
    conn.close()

    texts, labels = [], []
    # Placeholder: For this example, we'll need a real labeling strategy.
    # We are assuming all stored messages are 'positive' (label 1) for demonstration.
    for row in rows:
        try:
            decrypted_text = decrypt_data(bytes.fromhex(row['encrypted_message']), ENCRYPTION_KEY).decode('utf-8')
            texts.append(decrypted_text)
            labels.append(1) # 1 for positive, 0 for negative
        except Exception as e:
            print(f"Could not decrypt row: {e}")
    
    if not texts:
        return None, None, None, None

    # Split data into 80% training and 20% validation
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print(f"Loaded {len(texts)} total records.")
    print(f"Training set: {len(X_train)} records | Validation set: {len(X_test)} records")
    return X_train, X_test, y_train, y_test

# --- 2. Load Model (Improved or Base) --- #
def load_model():
    """Load the saved improved model if available, otherwise load base model."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Try to load saved improved model
    if os.path.exists(SAVED_MODEL_PATH):
        try:
            print(f"🔄 Loading improved federated model from: {SAVED_MODEL_PATH}")
            saved_state_dict = torch.load(SAVED_MODEL_PATH, map_location='cpu')
            model.load_state_dict(saved_state_dict, strict=True)
            print("✅ Successfully loaded improved federated model!")
            
            # Check if metadata exists
            metadata_path = os.path.join(os.path.dirname(SAVED_MODEL_PATH), "model_metadata.txt")
            if os.path.exists(metadata_path):
                print("📋 Model metadata:")
                with open(metadata_path, 'r') as f:
                    for line in f:
                        print(f"   {line.strip()}")
                        
        except Exception as e:
            print(f"⚠️  Error loading saved model: {e}")
            print(f"🔄 Falling back to base model: {MODEL_NAME}")
    else:
        print(f"📦 No saved model found. Using base model: {MODEL_NAME}")
        print(f"   (Improved model will be available after first FL training session)")
    
    return model, tokenizer

# --- 2. Define Flower Client --- #
class MentalHealthClient(fl.client.NumPyClient):
    def __init__(self, model, tokenizer, X_train, y_train, X_test, y_test):
        self.model = model
        self.tokenizer = tokenizer
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Set model parameters from the server
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Train on the training set
        inputs = self.tokenizer(self.X_train, padding=True, truncation=True, return_tensors="pt")
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(self.y_train))
        dataloader = DataLoader(dataset, batch_size=8)
        
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        for epoch in range(1): # Train for 1 epoch
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        
        print(f"Local training finished on {len(self.X_train)} samples.")
        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local validation set."""
        print(f"Evaluating model on {len(self.X_test)} validation samples.")
        # Set model parameters from the server
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Evaluate on the validation set
        inputs = self.tokenizer(self.X_test, padding=True, truncation=True, return_tensors="pt")
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(self.y_test))
        dataloader = DataLoader(dataset, batch_size=8)

        self.model.eval()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        accuracy = correct / total
        loss = total_loss / len(dataloader)
        print(f"Evaluation finished: Accuracy={accuracy:.4f}, Loss={loss:.4f}")
        return loss, len(self.X_test), {"accuracy": accuracy}

# --- 4. Start the Client --- #
if __name__ == "__main__":
    print("🧠 Federated Mental Health Client - Evaluation Enabled")
    print("=" * 60)

    # Load model and tokenizer (improved or base)
    model, tokenizer = load_model()

    # Load and split local data
    X_train, X_test, y_train, y_test = load_data()

    if not X_train:
        print("❌ No data available for training. Please use the chatbot to generate some data.")
    else:
        print(f"🚀 Starting client with {len(X_train)} training examples and {len(X_test)} validation examples...")
        # Create and start the Flower client
        client = MentalHealthClient(model, tokenizer, X_train, y_train, X_test, y_test)
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
        print("✅ Client finished its participation.")
