import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from graph.state import WealthManagerState

# 1. Model Setup
# We use ProsusAI/finbert as it is pre-trained on financial corpora [cite: 9]
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def get_finbert_score(text: str) -> float:
    """
    Analyzes text and returns a score between -1 (Bearish) and 1 (Bullish).
    """
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply softmax to get probabilities for [Positive, Negative, Neutral]
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
    
    # Calculate a single weighted sentiment score
    # FinBERT labels: 0: Positive, 1: Negative, 2: Neutral
    sentiment_score = probs[0] - probs[1]
    return float(sentiment_score)

def sentiment_node(state: WealthManagerState):
    """
    The LangGraph node for Sentiment Analysis.
    """
    print("--- 🔍 AGENT: SENTIMENT ANALYSIS (FinBERT) ---")
    
    # Extract the last user message/headline from the state
    latest_message = state["messages"][-1] if state["messages"] else ""
    
    # Run the specialized model
    score = get_finbert_score(latest_message)
    
    # Determine a label for the qualitative analysis report [cite: 17]
    label = "Bullish" if score > 0.2 else "Bearish" if score < -0.2 else "Neutral"
    
    # Update the shared state 
    return {
        "sentiment_score": score,
        "messages": [f"FinBERT Sentiment: {label} (Score: {score:.2f})"]
    }