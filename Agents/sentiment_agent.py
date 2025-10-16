import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from openai import OpenAI
import os

# Load Hugging Face emotion model
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# OpenAI client
#add api key here
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Original Hugging Face labels (from model card)
HF_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Mapping HF emotions → project categories
CATEGORY_MAP = {
    "anger": "Angry",
    "disgust": "Frustrated",
    "fear": "Frustrated",
    "sadness": "Frustrated",
    "joy": "Satisfied",
    "neutral": "Neutral",
    "surprise": "Neutral"   # treat surprise as neutral unless flagged otherwise
}

def classify_with_hf(text, threshold=0.5):
    """Classify sentiment using Hugging Face model and map to project categories."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    max_idx = int(np.argmax(probs))
    hf_label = HF_EMOTIONS[max_idx]
    mapped_label = CATEGORY_MAP[hf_label]
    max_prob = probs[max_idx]

    if max_prob >= threshold:
        return mapped_label, max_prob
    else:
        return None, max_prob  # uncertain → fallback to LLM

def classify_with_llm(text):
    """Fallback: classify sentiment using OpenAI GPT."""
    prompt = (
        "You are a customer support sentiment analyzer.\n"
        "Classify the following ticket into one of: Angry, Frustrated, Neutral, Satisfied.\n"
        "Respond with ONLY the category name.\n\n"
        f"Ticket: {text}"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )
    return response.choices[0].message.content.strip()

def classify_sentiment(text, threshold=0.5):
    """Hybrid classifier: Hugging Face first, then fallback to LLM if uncertain."""
    hf_label, prob = classify_with_hf(text, threshold)
    if hf_label:
        return {"method": "HuggingFace", "label": hf_label, "confidence": prob}
    else:
        llm_label = classify_with_llm(text)
        return {"method": "LLM-Fallback", "label": llm_label, "confidence": prob}
