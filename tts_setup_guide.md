# Groq TTS Setup Guide

## 🎯 Problem
The `playai-tts` model requires accepting Play.ht's Terms of Service before use.

## 🔧 Solution Steps

### Step 1: Accept Terms of Service
1. Go to [Groq Console](https://console.groq.com/)
2. Navigate to **Playground**
3. Look for **Text to Speech** or **TTS models**
4. Find `playai-tts` model
5. **Accept the Play.ht Terms of Service**
6. Wait for approval (usually instant)

### Step 2: Test TTS Access
Run our test script to verify access:
```bash
source venv/bin/activate
python3 test_groq_tts_fixed.py
```

### Step 3: Update Pipeline
Once TTS is working, the pipeline will automatically use it.

## 📋 Available Voices (19 English)

| Voice Name | Description |
|------------|-------------|
| Arista-PlayAI | Professional female voice |
| Atlas-PlayAI | Deep male voice |
| Basil-PlayAI | Warm male voice |
| Briggs-PlayAI | Strong male voice |
| Calum-PlayAI | Friendly male voice |
| Celeste-PlayAI | Clear female voice |
| Cheyenne-PlayAI | Natural female voice |
| Chip-PlayAI | Casual male voice |
| Cillian-PlayAI | Irish accent male |
| Deedee-PlayAI | Sweet female voice |
| Fritz-PlayAI | German accent male |
| Gail-PlayAI | Mature female voice |
| Indigo-PlayAI | Unique female voice |
| Mamaw-PlayAI | Southern female voice |
| Mason-PlayAI | Professional male voice |
| Mikail-PlayAI | Russian accent male |
| Mitch-PlayAI | Casual male voice |
| Quinn-PlayAI | Neutral voice |
| Thunder-PlayAI | Powerful male voice |

## 💰 Pricing
- **$50.00 per million characters**
- **20,000 characters per $1**
- Very cost-effective for most use cases

## 🚀 Quick Test
Once terms are accepted, test with:
```python
from groq import Groq

client = Groq(api_key="your_api_key")

response = client.audio.speech.create(
    model="playai-tts",
    voice="Mason-PlayAI",
    input="Hello! This is a test.",
    response_format="wav"
)

response.write_to_file("test.wav")
```

## 📚 Documentation Links
- [Groq TTS Documentation](https://console.groq.com/docs/text-to-speech)
- [PlayAI TTS Model Card](https://console.groq.com/docs/model/playai-tts)
- [Groq Playground](https://console.groq.com/playground) 