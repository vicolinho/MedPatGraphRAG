import json
import os
from datetime import datetime

import requests
from dotenv import load_dotenv
from langchain_community.chat_models import openai
from openai.types import ReasoningEffort
from pydantic import SecretStr
import openai
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings




def llm_query(api_url, api_key, prompt, model):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    payload = {
        "model": model, #miniMax, kimi
        "messages": [{"role": "user", "content": prompt}],
        # "max_tokens": 150,
        "temperature": 0.7,
        # "stop_sequences": ["\n"]  # Optional: sequences where generation should stop
        "response_format" : {"type": "json_object"}
    }
    response = requests.post(
        api_url,  # f-string for host
        headers=headers,
        data=json.dumps(payload),  # Use ollama_model
        stream=True,
        timeout=120  # Give model time to respond
    )
    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
    return response
