import os
import openai
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
import time
import elevenlabs
import time
import subprocess
from elevenlabs import set_api_key
from pydub import AudioSegment


load_dotenv()

elevenlabs.set_api_key("f940d1f2e759f922d927d14f7133bf5e")

app = FastAPI()

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", max_new_tokens=1500,temperature = 0.02
)

# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
)

qa_tmpl_str = (
    """
    Context information is below.\
    ---------------------\
    Context: {context_str}\
    Given the context information and not prior knowledge, \
    If the context mentions step-by-step information in the given context, please present the response in the following format:

    Step 1: [Step 1 Text]
    Step 2: [Step 2 Text]
    ...
    Step N: [Step N Text]
    don't add any unecessary steps or knowledge that does not exist in context.

    Ensure that the response adheres strictly to the step-wise structure if steps are present in the context. 
    If the information is not explicitly laid out in steps, present the answer in a paragraph or a format 
    that aligns with the context information .
    Query: {query_str}\
    Answer: 
    """
)
qa_tmpl = PromptTemplate(qa_tmpl_str)

query_engine = index.as_query_engine(
     multi_modal_llm=openai_mm_llm, text_qa_template=qa_tmpl
)

def input_user_guery(query_str):
    response = query_engine.query(query_str)
    return response

def response_to_list(response):
    steps_pattern = re.compile(r"Step \d+: .+")
    steps_list = steps_pattern.findall(str(response))
    
    # Wrap each step in the desired format
    formatted_steps = [f"{step}" for step in steps_list]
    
    return formatted_steps

model = FlagModel('BAAI/bge-large-en-v1.5', 
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")

def load_image(url_or_path):
    try:
        img = Image.open(url_or_path)
        return img
    except FileNotFoundError:
        return None
    

def cosine_similarity(vec1, vec2):
    # Compute the dot product of vec1 and vec2
    dot_product = np.dot(vec1, vec2)

    # Compute the L2 norm of vec1 and vec2
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return similarity

data = pd.read_excel("data/bank_app_data_embedding_3.xlsx")

def convert_embedding(x):
    try:
        return np.array(json.loads(x))
    except (json.JSONDecodeError, TypeError):
        return np.nan  # or any other value to represent missing data

# Apply the function only to non-empty values in the 'embedding' column
data['embedding'] = data['embedding'].apply(lambda x: convert_embedding(x) if x else np.nan)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def top_products(user_input):
    user_embedding=model.encode(user_input)
    data['scores']=None
    for i,row in data.iterrows():
        data.at[i,'scores']=cosine_similarity(user_embedding,row['embedding'])
    data['scores'] = pd.to_numeric(data['scores'], errors='coerce')
    top_product=data.nlargest(1,'scores') 
    if pd.notna(top_product['image_path'].iloc[0]):
        image_path = top_product['image_path'].iloc[0]
        base64_image = encode_image(image_path)

        return {'image_path' : image_path}
    




def elevenlabaudio(response):
    response_text = str(response)  # Convert the response object to a string
    assert isinstance(response_text, str), f"Expected response_text to be a string, but got {type(response_text)}"

    voice = elevenlabs.Voice(
        voice_id="EXAVITQu4vr4xnSDxMaL",
    )
    audio = elevenlabs.generate(
        text=response_text,
        voice=voice
    )
    
    # Generate a unique filename based on timestamp
    timestamp = int(time.time())
    mp3_file_name = f"audios/audio_{timestamp}.mp3"
    wav_file_name = f"audios/audio_{timestamp}.wav"
    json_output_path = f"audios/audio_json_{timestamp}.json"
    

    # Save audio file as MP3
    elevenlabs.save(audio, mp3_file_name)

    # Convert MP3 to WAV using pydub
    audio = AudioSegment.from_mp3(mp3_file_name)
    audio.export(wav_file_name, format="wav")
    run_rhubarb_lip_sync(wav_file_name, json_output_path)

    audio_path = 'audios'
    if audio_path is not None:
        # Encode WAV file to base64
        with open(wav_file_name, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
        with open(json_output_path,"rb") as lipsync_json_file:
            lipsync_json = lipsync_json_file.read()
            parsed_data = json.loads(lipsync_json)

        
        return encoded_audio,parsed_data
    else:
        # Handle the case where audio_path is None (e.g., save failed)
        return None

def run_rhubarb_lip_sync(wav_file_path, json_output_path):
    try:
    # Construct the Rhubarb Lip Sync command
        command = [
            './bin/Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb',
            '-f', 'json',
            '-o', f'{json_output_path}',
            f'{wav_file_path}',
            '-r', 'phonetic',
        ]
        
        # Execute the command
        subprocess.run(command, check=True)
        
        print(f"Lip sync done for message {json_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during Rhubarb Lip Sync: {e}")



        


@app.get("/get_response/{user_guery}")
def get_response(user_guery:str):
    start_time = time.time()

    response = input_user_guery(user_guery)

    encoded_audio,parsed_data  = elevenlabaudio(response)

    print(response)
    steps = response_to_list(str(response))
    list_of_step_image = []
    if steps:
        for step in steps:
            inp = step
            print('nnnn' + step)
            image_data = top_products(inp)
            
            if image_data:
                list_of_step_image.append({"response": step, "image": image_data, "audio": encoded_audio, "lipsync":parsed_data})
            else:
                list_of_step_image.append({"response": step,  "audio": encoded_audio,"lipsync":parsed_data})

    else:
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(elapsed_time) 
        return {"response": str(response), "image": None,  "audio": encoded_audio,"lipsync":parsed_data}
        
    
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(elapsed_time)
    return list_of_step_image