# Groq Pipeline

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
python3 groq_pipeline_fixed_queue.py
```

## 🔧 How It Works

The pipeline implements a complete speech-to-speech conversation system:

1. **Speech-to-Text**: Real-time audio capture using Deepgram WebSocket API
2. **LLM Processing**: Intelligent text processing using Groq's llama3-8b-8192 model
3. **Text-to-Speech**: High-quality voice synthesis using Groq's PlayAI TTS
4. **Echo Prevention**: Smart microphone management to prevent feedback loops
5. **Queue Management**: Optimized audio queue handling for smooth operation

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
- **Echo Prevention**: Prevents AI from responding to its own speech
- **Queue Management**: Optimized audio handling for smooth operation
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
Edit `groq_pipeline_fixed_queue.py` line ~170:
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