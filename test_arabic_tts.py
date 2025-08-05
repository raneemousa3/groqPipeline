#!/usr/bin/env python3
"""
Test script for Arabic TTS functionality
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

print("🔊 Testing Arabic TTS...")

try:
    # Initialize Groq client
    client = Groq(api_key=GROQ_API_KEY)
    
    # Test Arabic TTS models
    arabic_tts_configs = [
        {
            "model": "playai-tts-arabic",
            "voice": "Ahmad-PlayAI",
            "text": "مرحباً! كيف حالك اليوم؟",
            "description": "Arabic TTS - Ahmad voice"
        },
        {
            "model": "playai-tts-arabic", 
            "voice": "Amira-PlayAI",
            "text": "أهلاً وسهلاً بك! كيف يمكنني مساعدتك؟",
            "description": "Arabic TTS - Amira voice"
        },
        {
            "model": "playai-tts-arabic",
            "voice": "Khalid-PlayAI", 
            "text": "مرحباً! أنا مساعدك الذكي.",
            "description": "Arabic TTS - Khalid voice"
        },
        {
            "model": "playai-tts-arabic",
            "voice": "Nasser-PlayAI", 
            "text": "أهلاً! كيف يمكنني مساعدتك اليوم؟",
            "description": "Arabic TTS - Nasser voice"
        }
    ]
    
    for config in arabic_tts_configs:
        try:
            print(f"\n🔍 Testing: {config['description']}")
            print(f"   Model: {config['model']}")
            print(f"   Voice: {config['voice']}")
            print(f"   Text: {config['text']}")
            
            # Generate unique filename
            filename = f"arabic_test_{uuid.uuid4().hex[:8]}.wav"
            
            # Test Arabic TTS generation
            response = client.audio.speech.create(
                model=config["model"],
                voice=config["voice"],
                input=config["text"],
                response_format="wav"
            )
            
            # Save audio file
            response.write_to_file(filename)
            
            print(f"✅ Arabic TTS successful!")
            print(f"📁 Audio file created: {filename}")
            
            # Play the audio
            print(f"🔊 Playing Arabic audio...")
            os.system(f"afplay {filename}")
            
            # Clean up after playing
            import time
            time.sleep(3)
            if os.path.exists(filename):
                os.remove(filename)
                print(f"🗑️  Cleaned up {filename}")
            
            # If we get here, Arabic TTS is working!
            print(f"🎉 Arabic TTS is working with {config['model']}!")
            break
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Failed: {error_msg}")
            
            if "terms acceptance" in error_msg.lower():
                print("💡 You need to accept terms for Arabic TTS models!")
                print("🔗 Go to: https://console.groq.com/playground")
                print("   Look for Arabic TTS models and accept the terms")
            elif "model not found" in error_msg.lower():
                print("💡 This Arabic model might not be available in your account")
            else:
                print("💡 Unknown error - check your API key and permissions")
    
except Exception as e:
    print(f"❌ Error in Arabic TTS test: {e}")

print("\n📚 Documentation:")
print("- Groq Arabic TTS: https://console.groq.com/docs/text-to-speech")
print("- Available Arabic voices: Ahmad, Amira, Khalid, Nasser")
print("- Model: playai-tts-arabic") 