# groqPipeline

A real-time speech-to-speech pipeline using Deepgram STT, Groq LLM, and Groq TTS.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file with your API keys:
```bash
DEEPGRAM_API_KEY=your_deepgram_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Pipeline
```bash
python groq_pipeline.py
```

## 🔧 How It Works

# ─────────────────────────────────────
# 1. Initialization
# ─────────────────────────────────────

load_env_variables()                        # Load .env variables like DEEPGRAM_API_KEY, GROQ_API_KEY
init_deepgram_ws_connection()              # Set up WebSocket connection to Deepgram STT
init_groq_api_clients()                    # Set up Groq LLM and TTS client(s)
create_audio_queue()                       # Create a queue to buffer audio chunks

# ─────────────────────────────────────
# 2. Start STT Loop (Mic → Text)
# ─────────────────────────────────────

async def stream_audio_to_deepgram():
    while mic_is_open:
        audio_chunk = get_audio_from_microphone()         # Capture small audio slice from mic
        send_audio_to_deepgram(audio_chunk)               # Send it to Deepgram via WebSocket

async def receive_text_from_deepgram():
    while connection_is_open:
        transcription = await receive_transcript()        # Wait for a transcript from Deepgram
        if transcription.is_final:
            handle_transcribed_text(transcription.text)   # Pass it to the next stage

# ─────────────────────────────────────
# 3. LLM Stage (Text → Meaning → Response)
# ─────────────────────────────────────

def handle_transcribed_text(text):
    prompt = build_prompt_from_text(text)                 # Add system instruction/context to user input
    response = query_groq_llm(prompt)                     # Send prompt to Groq and get assistant reply
    handle_llm_response(response)                         # Pass reply to TTS

def build_prompt_from_text(user_input):
    return f"You are a helpful assistant.\nUser: {user_input}\nAssistant:"  # Customize prompt

def query_groq_llm(prompt):
    return call_groq_chat_completion(prompt)              # Call Groq LLM completion endpoint

# ─────────────────────────────────────
# 4. TTS Stage (Text → Voice)
# ─────────────────────────────────────

def handle_llm_response(text_response):
    audio_file = convert_text_to_speech(text_response)    # Generate voice from response text
    play_audio(audio_file)                                # Play the audio back to the user

def convert_text_to_speech(text):
    filename = generate_uuid_filename()                   # Create unique name like abc123.wav
    save_groq_voice_to_file(text, filename)               # Use Groq TTS API to save speech to file
    return filename

def play_audio(file_path):
    play(file_path)                                       # Play the audio (e.g., with pydub or playsound)
    delete(file_path)                                     # Optional: delete audio file after playback

# ─────────────────────────────────────
# 5. Run the App (Main Loop)
# ─────────────────────────────────────

async def main():
    await run_in_parallel(
        stream_audio_to_deepgram(),                       # Start capturing and sending audio
        receive_text_from_deepgram()                      # Start receiving transcribed text
    )

if __name__ == "__main__":
    asyncio.run(main())                                   # Run both STT tasks together

## 🏗️ Architecture

### Core Components

1. **ConversationContext**: Manages conversation history and context
2. **PromptEngineer**: Builds intelligent prompts with conversation context
3. **GroqPipeline**: Main orchestrator that connects all components

### Flow Diagram
```
🎤 Microphone → 🔊 Audio Capture → 📡 Deepgram STT → 📝 Text
                                                           ↓
🎧 Audio Output ← 🔊 TTS Generation ← 🤖 Groq LLM ← 🧠 Prompt Engineering
```

## 🔑 Features

- **Real-time Processing**: Minimal latency between speech input and response
- **Context Awareness**: Maintains conversation history for better responses
- **Intelligent Prompting**: Dynamic prompt generation based on conversation context
- **Error Handling**: Graceful handling of API failures and audio issues
- **Clean Audio Management**: Automatic cleanup of temporary audio files

## 📋 Requirements

- Python 3.8+
- Deepgram API key
- Groq API key
- Microphone access
- Speakers/headphones

## 🛠️ Customization

### Changing the LLM Model
Edit `groq_pipeline.py` line ~150:
```python
response = groq_llm.chat.completions.create(
    model="llama3-8b-8192",  # Change to your preferred model
    # ...
)
```

### Modifying the System Prompt
Edit the `system_prompt` in the `PromptEngineer` class:
```python
self.system_prompt = """Your custom system prompt here"""
```

### Adjusting Audio Settings
Modify the audio configuration variables:
```python
SAMPLE_RATE = 16_000  # Audio sample rate
CHUNK_MS = 100        # Audio chunk size in milliseconds
```

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'groq'"**: Make sure you're in the virtual environment
2. **"API key not found"**: Check your `.env` file has the correct API keys
3. **Audio not playing**: Ensure your system audio is working and not muted
4. **Microphone not detected**: Check microphone permissions in system settings

### Debug Mode
Add debug prints by modifying the pipeline code or set environment variables for more verbose logging.

## 📝 License

This project is open source and available under the MIT License. 