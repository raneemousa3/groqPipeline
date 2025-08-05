#!/usr/bin/env python3
"""
Test script for Groq TTS
"""

import os
import sys
import uuid
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv('nAgent/backend/api_keys/.env')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found!")
    sys.exit(1)

print("🔊 Testing Groq TTS...")

try:
    # Initialize Groq client
    client = Groq(api_key=GROQ_API_KEY)
    
    # Generate unique filename
    filename = f"test_tts_{uuid.uuid4().hex[:8]}.wav"
    
    # Test TTS generation with a different model
    response = client.audio.speech.create(
        model="llama3.2-70b-4096",  # Try a different model
        voice="alloy",
        input="Hello! This is a test of the text to speech system.",
        response_format="wav"
    )
    
    # Save audio file
    response.write_to_file(filename)
    
    print(f"✅ Groq TTS test successful!")
    print(f"📁 Audio file created: {filename}")
    print(f"🔊 Playing audio file...")
    
    # Play the audio
    import os
    os.system(f"afplay {filename}")
    
    # Clean up after a few seconds
    import time
    time.sleep(3)
    if os.path.exists(filename):
        os.remove(filename)
        print(f"🗑️  Cleaned up {filename}")
    
except Exception as e:
    print(f"❌ Groq TTS test failed: {e}")
    print("💡 Note: You may need to accept terms for TTS models at https://console.groq.com/playground")
    sys.exit(1) 