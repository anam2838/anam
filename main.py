import whisper
import sounddevice as sd
import requests
import os
from gtts import gTTS
# Load the Whisper model
model = whisper.load_model("base.en")

def record_audio(duration=5, sample_rate=16000):
    """Record audio using the microphone."""
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32') 
    sd.wait()  # Wait until the recording is finished
    return audio.flatten()
    
def transcribe_audio(audio):
    """Transcribe recorded audio to text using Whisper."""
    print("Transcribing audio...")
    try:
        result = model.transcribe(audio)
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
        pass

def query_ollama(prompt):
    """Send the transcribed text to the locally running Ollama model and get the response."""
    url = "http://localhost:11434/api/generate"  # Update to the correct API endpoint if necessary
    data = {
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False  # Non-streaming for simplicity
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json().get("response", "No response received.")
        else:
            print(f"Error querying Ollama: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return None

def text_to_speech(text):
    """Convert the model's response to speech using gTTS and play it."""
    print("Converting text to speech...")
    try:
        tts = gTTS(text, lang="en")
        tts.save("response.mp3")
        os.system("mpg123 response.mp3")  # Ensure mpg123 is installed
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")

def main():
    """Main function to handle a conversation loop."""
    print("Voice Assistant Initialized. You can speak now...")
    for i in range(5):
        print(f"\n--- Conversation {i + 1} ---")
        
        # Record audio
        audio = record_audio()
        
        # Transcribe audio
        transcribed_text = transcribe_audio(audio)
        if not transcribed_text:
            print("No transcription available. Skipping...")
            continue
        
        print("Transcribed Text:", transcribed_text)
        
        # Query Ollama
        response = query_ollama(transcribed_text)
        if response:
            print("Model Response:", response)
            text_to_speech(response)
        else:
            print("Failed to get a response from the model.")

    print("\nConversation Ended. Thank you!")

if __name__ == "__main__":
    main()
