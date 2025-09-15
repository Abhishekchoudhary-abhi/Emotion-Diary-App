# src/text_sentiment.py
from transformers import pipeline

# Load the model once when the script starts for efficiency
print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline("sentiment-analysis")
print("Model loaded.")

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text.
    Returns the sentiment label ('POSITIVE' or 'NEGATIVE').
    """
    try:
        result = sentiment_pipeline(text)
        return result[0]['label']
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "unknown"

# This block is for testing the script directly
if __name__ == "__main__":
    test_text = "Today was a fantastic day, full of joy and success."
    sentiment = analyze_sentiment(test_text)
    print(f"\nText: '{test_text}'")
    print(f"Detected Sentiment: {sentiment}")