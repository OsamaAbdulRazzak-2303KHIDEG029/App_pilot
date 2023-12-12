import elevenlabs
from dotenv import load_dotenv
import os

load_dotenv()

responses = ["Hello", "How are you?", "Goodbye"]

def elevenlabaudio(responses):
    voice = elevenlabs.Voice(
        voice_id="EXAVITQu4vr4xnSDxMaL",
    )

    for i, response in enumerate(responses):
        audio = elevenlabs.generate(
            text=response,
            voice=voice,
        )
        
        # Save audio file with a numbered filename
        file_name = f"audios/audio_{i}.mp3"
        
        if os.path.exists(file_name):
            os.remove(file_name)

        elevenlabs.save(audio, file_name)
        print(f"Audio file saved: {file_name}")

    return elevenlabs.play(audio)

# Call the function
elevenlabaudio(responses)
