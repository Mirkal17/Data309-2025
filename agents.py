# Agents
from Agents.classifier_agent import *
from Agents.sentiment_agent import *
from Agents.escalation_agent import *
import openai
import os

# Set API key
# add key in this line
openai.api_key = os.getenv("OPENAI_API_KEY")

def evaluate_ticket(ticket_description):
    global messages
    classification = ticket_classification(ticket_description)
    sentiment = classify_sentiment(ticket_description)
    escalation_answer = escalation(sentiment)
    print(f"Classification: {classification}")
    print(f"Sentiment: {sentiment['label']}")
    print(f"Escalation: {escalation_answer}")
    final_message = create_final_response(ticket_description, classification, sentiment, escalation_answer) 
    return final_message

def create_final_response(ticket_description, classification, sentiment, escalation): 
    if escalation == "solution": 
        prompt_text = f"""The ticket category is {classification}. The customer is having issues with {ticket_description}. The customer is feeling {sentiment} so reply accordingly.""" 
        response = openai.chat.completions.create( 
            model="gpt-3.5-turbo", 
            messages=[ {"role": "system", "content": """You are a helpful customer support agent that gives a solution to an issue. Tailor your reply depending on how the customer is feeling."""}, 
            {"role": "user", "content": prompt_text} ] ) 
        answer = response.choices[0].message.content 
        return answer 
    else: 
        return f"The ticket regarding {classification} cannot be resolved through this chatbot. Please contact 'Example@gmail.com'"