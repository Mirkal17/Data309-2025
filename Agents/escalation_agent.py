from openai import OpenAI
import os

def escalation(sentiment):
    unhappy_list = ["Angry", "Disgust", "Fear", "Frustrated"]
    if sentiment['label'] in unhappy_list:
        return 'human'
    else:
        return 'solution'


