#!/usr/bin/env python3
"""
Test script for Groq TTS using correct models from documentation
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

print("🔊 Testing Groq TTS with correct models...")

try:
    # Initialize Groq client
    client = Groq(api_key=GROQ_API_KEY)
    
    # Test the correct TTS models from documentation
    tts_configs = [
        {
            "model": "playai-tts",
            "voice": "Mason-PlayAI",
            "description": "English TTS - Mason voice"
        },
        {
            "model": "playai-tts", 
            "voice": "Fritz-PlayAI",
            "description": "English TTS - Fritz voice"
        },
        {
            "model": "playai-tts-arabic",
            "voice": "Ahmad-PlayAI", 
            "description": "Arabic TTS - Ahmad voice"
        }
    ]
    
    for config in tts_configs:
        try:
            print(f"\n🔍 Testing: {config['description']}")
            print(f"   Model: {config['model']}")
            print(f"   Voice: {config['voice']}")
            
            # Generate unique filename
            filename = f"test_tts_{uuid.uuid4().hex[:8]}.wav"
            
            # Test TTS generation
            response = client.audio.speech.create(
                model=config["model"],
                voice=config["voice"],
                input="Hello! This is a test of the text to speech system.",
                response_format="wav"
            )
            
            # Save audio file
            response.write_to_file(filename)
            
            print(f"✅ TTS successful!")
            print(f"📁 Audio file created: {filename}")
            
            # Play the audio
            print(f"🔊 Playing audio...")
            os.system(f"afplay {filename}")
            
            # Clean up after playing
            import time
            time.sleep(3)
            if os.path.exists(filename):
                os.remove(filename)
                print(f"🗑️  Cleaned up {filename}")
            
            # If we get here, TTS is working!
            print(f"🎉 TTS is working with {config['model']}!")
            break
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Failed: {error_msg}")
            
            if "terms acceptance" in error_msg.lower():
                print("💡 You need to accept terms for TTS models!")
                print("🔗 Go to: https://console.groq.com/playground")
                print("   Look for TTS models and accept the terms")
            elif "model not found" in error_msg.lower():
                print("💡 This model might not be available in your account")
            else:
                print("💡 Unknown error - check your API key and permissions")
    
except Exception as e:
    print(f"❌ Error in TTS test: {e}")

print("\n📚 Documentation:")
print("- Groq TTS: https://console.groq.com/docs/text-to-speech")
print("- Available voices: English (19), Arabic (4)")
print("- Models: playai-tts, playai-tts-arabic") 