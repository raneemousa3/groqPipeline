#!/usr/bin/env python3
"""
Alternative TTS Services for the Groq Pipeline
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv('nAgent/backend/api_keys/.env')

print("🔊 TTS Alternatives for Groq Pipeline")
print("=" * 50)

# Option 1: ElevenLabs TTS
print("\n1️⃣ **ElevenLabs TTS**")
print("   ✅ High quality, natural voices")
print("   ✅ Easy to use")
print("   ✅ Free tier available")
print("   📝 Setup:")
print("      - Sign up at https://elevenlabs.io")
print("      - Get API key")
print("      - Add ELEVENLABS_API_KEY to .env")

# Option 2: OpenAI TTS
print("\n2️⃣ **OpenAI TTS**")
print("   ✅ High quality, multiple voices")
print("   ✅ Fast and reliable")
print("   ✅ Good integration")
print("   📝 Setup:")
print("      - Get OpenAI API key")
print("      - Add OPENAI_API_KEY to .env")

# Option 3: Google Cloud TTS
print("\n3️⃣ **Google Cloud TTS**")
print("   ✅ Enterprise-grade")
print("   ✅ Many languages")
print("   ✅ Good documentation")
print("   📝 Setup:")
print("      - Google Cloud account")
print("      - Enable TTS API")
print("      - Add GOOGLE_CLOUD_CREDENTIALS to .env")

# Option 4: Azure TTS
print("\n4️⃣ **Azure TTS**")
print("   ✅ Microsoft's solution")
print("   ✅ Neural voices")
print("   ✅ Good for enterprise")
print("   📝 Setup:")
print("      - Azure account")
print("      - Cognitive Services")
print("      - Add AZURE_SPEECH_KEY to .env")

# Option 5: Local TTS (pyttsx3)
print("\n5️⃣ **Local TTS (pyttsx3)**")
print("   ✅ No API keys needed")
print("   ✅ Works offline")
print("   ✅ Free")
print("   📝 Setup:")
print("      - pip install pyttsx3")
print("      - Uses system voices")

print("\n" + "=" * 50)
print("💡 **Recommendation:**")
print("   For quick setup: ElevenLabs")
print("   For production: OpenAI or Google Cloud")
print("   For offline: pyttsx3")

print("\n🔧 **Next Steps:**")
print("   1. Choose a TTS service")
print("   2. Get API key")
print("   3. Update pipeline code")
print("   4. Test integration") 