
import os
from openai import OpenAI
from dotenv import load_dotenv
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.prompts import PromptTemplate
import re
import pandas as pd
import numpy as np
from PIL import Image
from FlagEmbedding import FlagModel
import json
from fastapi import FastAPI
from llama_index.vector_stores import ChromaVectorStore
import chromadb
from llama_index import VectorStoreIndex
import base64
import requests
import time
import elevenlabs
import time
from openai import chat
import subprocess
from elevenlabs import set_api_key
from openai import OpenAI
import openai


load_dotenv()


# Load the OpenAI API key from the environment variable
openai_api_key = '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'

# # Check if the API key is available
# if not openai_api_key:
#     raise ValueError("OpenAI API key is missing. Set the OPENAI_API_KEY environment variable.")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=openai_api_key)
# GPT_MODEL = "gpt-3.5-turbo-0613"

# def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": "Bearer " + openai.api_key,
#     }
#     json_data = {"model": model, "messages": messages}
#     if tools is not None:
#         json_data.update({"tools": tools})
#     if tool_choice is not None:
#         json_data.update({"tool_choice": tool_choice})
#     try:
#         response = requests.post(
#             "https://api.openai.com/v1/chat/completions",
#             headers=headers,
#             json=json_data,
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e



# openai_mm_llm = OpenAIMultiModal(
#     model="gpt-4-vision-preview", max_new_tokens=1500,temperature = 0.02
# )

# load from disk
# db2 = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection = db2.get_or_create_collection("quickstart")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# index = VectorStoreIndex.from_vector_store(
#     vector_store,
# )

custom_functions = [
    {
        "type": "function",
        "function": {
            "name": 'parse_text_facial_animation',
            "description": 'parses text response, and facial expressions according to the tone of text and animation type into json.',
            "parameters": {
                "type": 'object',
                "properties": {
                    "text": {
                        "type": 'string',
                        "description": 'response from the given prompt.'
                    },
                    "facialExpression": {
                        "type": 'string',
                        "enum": ['smile', 'sad', 'angry', 'surprised', 'funnyFace', 'default'],
                        "description": 'facial expressions based on the tone of response.'
                    },
                    "animation": {
                        "type": 'string',
                        "enum": ['Talking_0', 'Talking_1', 'Talking_2', 'Crying', 'Laughing', 'Rumba', 'Idle', 'Terrified', 'Angry'],
                        "description": 'animation type to display, select any from the Talking_0, Talking_1, Talking_2, if not sure about the type.'
                    },
                },
                "required": ["text", "facialExpression", "animation"],
            }
        },
    },
]

user_message = "Hello"

completion = client.chat.completions.create(model="gpt-3.5-turbo",
max_tokens=1000,
temperature=0.6,
messages=[
    {
        "role": "system",
        "content": """
        -Don't make assumptions about what values to plug into functions
        """
    },
    {
        "role": "user",
        "content": user_message or "Hello",
    },
],
tools=custom_functions,
tool_choice={"type": "function", "function": {"name": "parse_text_facial_animation"}})

tool_calls = completion['choices'][0]['message']['tool_calls']
arguments = json.loads(tool_calls[0]['function']['arguments'])

messages = arguments.get('messages', [arguments])
if isinstance(messages, dict) and 'messages' in messages:
    messages = messages['messages']

print(messages)