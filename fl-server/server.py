"""
Flower server for federated aggregation (FedAvg).

This script starts a Flower server and waits for clients to connect.
It orchestrates the training rounds and aggregates the results.
Now includes model persistence to save the final aggregated model.
"""

import flwr as fl
import os
import json
from datetime import datetime
import torch
import sys

# --- Add project root to sys.path --- #
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from flwr.common.parameter import parameters_to_ndarrays
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Model configuration
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_DIR = os.path.join(project_root, "saved_models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "federated_model.pt")

def get_evaluate_fn(model_name):
    """Return an evaluation function for server-side evaluation."""
    # Note: This function is not strictly necessary for this implementation
    # as we are relying on client-side evaluation, but it's good practice
    # to have a placeholder for potential server-side testing.
    def evaluate(server_round, parameters, config):
        return None, {}
    return evaluate

class ModelPersistenceStrategy(fl.server.strategy.FedAvg):
    """Custom strategy that saves the final aggregated model and logs accuracy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and store final parameters."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
            print(f"[Round {server_round}] Model parameters aggregated and stored.")
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results and log accuracy."""
        if not results:
            return None, {}
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"🎯 [Round {server_round}] Aggregated Accuracy: {accuracy_aggregated:.4f}")
        return aggregated_loss, {"accuracy": accuracy_aggregated}

    def save_final_model(self):
        """Save the final aggregated model to disk."""
        if self.final_parameters is not None:
            print("Training completed. Saving final aggregated model...")
            try:
                os.makedirs(MODEL_DIR, exist_ok=True)
                ndarrays = parameters_to_ndarrays(self.final_parameters)
                temp_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
                params_dict = zip(temp_model.state_dict().keys(), ndarrays)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                torch.save(state_dict, MODEL_SAVE_PATH)
                metadata = {"saved_at": str(datetime.now()), "model_name": MODEL_NAME}
                with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=4)
                print(f"✅ Final model saved to {MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"❌ Error saving model: {e}")

if __name__ == "__main__":
    # Define the custom strategy with model persistence and evaluation
    strategy = ModelPersistenceStrategy(
        min_available_clients=2,        # Wait for at least 2 clients to be connected
        min_fit_clients=2,            # Use at least 2 clients for training in each round
        min_evaluate_clients=2,       # Use at least 2 clients for evaluation
        evaluate_fn=get_evaluate_fn(MODEL_NAME), # Server-side evaluation function
    )

    # Start the Flower server
    print("Starting Flower server on 127.0.0.1:8080")
    print("Model persistence enabled - final model will be saved after training.")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3), # Run for 3 rounds of training
        strategy=strategy,
    )
    
    # Save the final model after training completes
    print("Training completed. Saving final aggregated model...")
    strategy.save_final_model()
    print("Server finished.")
