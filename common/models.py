"""
Shared model loading and utility functions (e.g., BERT/DistilBERT).
"""

# Shared model loading logic for Gemini, HuggingFace sentiment, and emotion models
from google import genai
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
    """
    Wrapper class for Google Gemini API using the modern google-genai SDK.
    Includes retry logic with exponential backoff for rate limiting.
    """
    def __init__(self, api_key: str, id: str, temperature: float = 0.2, **kwargs):
        # Initialize the client with API key using new google-genai SDK
        self.client = genai.Client(api_key=api_key)
        self.model_id = id
        self.temperature = temperature
        self.max_retries = 3
        self.initial_retry_delay = 2  # seconds
        print(f"[Gemini] ✅ Successfully initialized Gemini with model: {self.model_id}")

    def generate(self, prompt: str):
        """
        Generate content using Gemini model with retry logic for rate limits.
        
        Args:
            prompt: The prompt string to send to the model
            
        Returns:
            GeminiResponse: A wrapper object with the model's response
        """
        import time
        
        for attempt in range(self.max_retries):
            try:
                print(f"[Gemini] Attempt {attempt + 1}/{self.max_retries}")
                print(f"[Gemini] Sending prompt to {self.model_id}...")
                
                # Use the new google-genai SDK API
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=1024
                    )
                )
                
                print(f"[Gemini] ✅ Response received successfully on attempt {attempt + 1}")
                return GeminiResponse(response)
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate" in error_str or "quota" in error_str or "too many" in error_str
                
                if is_rate_limit and attempt < self.max_retries - 1:
                    # Calculate exponential backoff delay
                    delay = self.initial_retry_delay * (2 ** attempt)
                    print(f"[Gemini] ⚠️  Rate limit hit (429/TooManyRequests)")
                    print(f"[Gemini] Retrying in {delay} seconds (attempt {attempt + 1}/{self.max_retries})...")
                    time.sleep(delay)
                    continue
                else:
                    # Not a rate limit error or max retries reached
                    print(f"[Gemini] ❌ Error on attempt {attempt + 1}: {str(e)}")
                    import traceback
                    print(f"[Gemini] Traceback: {traceback.format_exc()}")
                    raise


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
        print(f"[GeminiResponse] Initializing with response: {type(response)}")
        
    @property
    def text(self) -> str:
        """
        Get the text content of the response.
        
        Returns:
            The text content as a string.
        """
        try:
            print(f"[GeminiResponse] Attempting to extract text...")
            print(f"[GeminiResponse] Response type: {type(self.raw_response)}")
            print(f"[GeminiResponse] Response attributes: {[attr for attr in dir(self.raw_response) if not attr.startswith('_')]}")
            
            # First, try direct text property (most common)
            if hasattr(self.raw_response, 'text') and self.raw_response.text:
                text_content = str(self.raw_response.text).strip()
                if text_content and text_content != "None":
                    print(f"[GeminiResponse] ✅ Got text from direct .text property")
                    return text_content
            
            # Check if response is blocked by safety filters
            if hasattr(self.raw_response, 'prompt_feedback'):
                feedback = self.raw_response.prompt_feedback
                print(f"[GeminiResponse] Prompt feedback present: {feedback}")
                if hasattr(feedback, 'block_reason') and feedback.block_reason:
                    print(f"[WARNING] Response blocked by safety filter: {feedback.block_reason}")
                    return f"[Content blocked by safety filter: {feedback.block_reason}]"
            
            # Check candidates (alternate structure)
            if hasattr(self.raw_response, 'candidates') and self.raw_response.candidates:
                print(f"[GeminiResponse] Found {len(self.raw_response.candidates)} candidate(s)")
                
                for idx, candidate in enumerate(self.raw_response.candidates):
                    finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
                    print(f"[GeminiResponse] Candidate {idx}: finish_reason = {finish_reason}")
                    
                    # Check for safety ratings
                    if hasattr(candidate, 'safety_ratings'):
                        for rating in candidate.safety_ratings:
                            print(f"[GeminiResponse] Safety rating: {rating}")
                    
                    # Try to extract content
                    if hasattr(candidate, 'content') and candidate.content:
                        content = candidate.content
                        if hasattr(content, 'parts') and content.parts:
                            texts = []
                            for part in content.parts:
                                if hasattr(part, 'text'):
                                    part_text = str(part.text).strip()
                                    if part_text and part_text != "None":
                                        texts.append(part_text)
                            if texts:
                                result = "".join(texts)
                                print(f"[GeminiResponse] ✅ Successfully extracted from candidates: {result[:100]}...")
                                return result
            
            # Fallback: check content property
            if hasattr(self.raw_response, 'content'):
                print(f"[GeminiResponse] Response has content property")
                content = self.raw_response.content
                if hasattr(content, 'parts'):
                    texts = []
                    for part in content.parts:
                        if hasattr(part, 'text'):
                            part_text = str(part.text).strip()
                            if part_text and part_text != "None":
                                texts.append(part_text)
                    if texts:
                        result = "".join(texts)
                        print(f"[GeminiResponse] ✅ Extracted from content.parts: {result[:100]}...")
                        return result
            
            # Last resort: dump everything for debugging
            print(f"[GeminiResponse] ❌ Could not extract text from response")
            print(f"[GeminiResponse] Full response repr: {repr(self.raw_response)[:500]}")
            print(f"[GeminiResponse] Converting to string...")
            fallback_text = str(self.raw_response)
            if fallback_text and fallback_text != "None" and len(fallback_text) > 10:
                print(f"[GeminiResponse] Using string fallback: {fallback_text[:100]}")
                return fallback_text
            
            return "[Unable to extract response text from Gemini API - please check logs]"
            
        except Exception as e:
            print(f"[GeminiResponse] ❌ Exception while extracting text: {str(e)}")
            import traceback
            print(f"[GeminiResponse] Traceback: {traceback.format_exc()}")
            return f"[Error extracting response: {str(e)}]"

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