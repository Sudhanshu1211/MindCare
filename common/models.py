"""
Shared model loading and utility functions (e.g., BERT/DistilBERT).
"""

# Shared model loading logic for Gemini, HuggingFace sentiment, and emotion models
import google.generativeai as genai
import torch
from typing import Dict, Any, Optional
from transformers import pipeline

# Determine device for model inference (GPU if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] Using device: {DEVICE}")

# HuggingFace Sentiment Pipeline Loader
_sentiment_pipeline = None
def get_sentiment_pipeline():
    """
    Initializes and returns a sentiment analysis pipeline.
    Handles truncation for long inputs.
    """
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=0 if DEVICE == "cuda" else -1,
            truncation=True  # Handle texts longer than the model's max length
        )
    return _sentiment_pipeline

# HuggingFace Emotion Pipeline Loader
_emotion_pipeline = None
def get_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        _emotion_pipeline = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion"
        )
    return _emotion_pipeline

class Gemini:
    def __init__(self, api_key='api_key', id='gemini-2.0-flash', temprature=0.2, **kwargs):
        self.api_key = api_key
        self.id = id
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            self.id,
            generation_config=genai.GenerationConfig(
                temperature=temprature,
                **kwargs
            )
        )
    
    def generate(self, prompt):
        """
        Generate content using Gemini model
        
        Args:
            prompt: The prompt string or object to send to the model
            
        Returns:
            GeminiResponse: A wrapper object with the model's response
        """
        response = self.model.generate_content([prompt])
        return GeminiResponse(response)


class GeminiResponse:
    """
    Wrapper class for Gemini responses to provide a consistent interface
    """
    
    def __init__(self, response):
        """
        Initialize with a raw Gemini response
        
        Args:
            response: The raw response from the Gemini model
        """
        self.raw_response = response
        
    @property
    def text(self) -> str:
        """
        Get the text content of the response
        
        Returns:
            The text content as a string
        """
        try:
            # Handle different response formats
            if hasattr(self.raw_response, "text"):
                return self.raw_response.text
            elif hasattr(self.raw_response, "parts"):
                return "".join(part.text for part in self.raw_response.parts)
            elif hasattr(self.raw_response, "candidates"):
                candidates = self.raw_response.candidates
                if candidates and len(candidates) > 0:
                    parts = candidates[0].content.parts
                    return "".join(part.text for part in parts)
            
            # Fallback: convert to string
            return str(self.raw_response)
            
        except Exception as e:
            # Return error message if parsing fails
            return f"Error extracting text from response: {str(e)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response to a dictionary
        
        Returns:
            Dictionary representation of the response
        """
        return {
            "text": self.text,
            # Add more properties as needed
            }