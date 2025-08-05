#!/usr/bin/env python3
"""
Guide for accepting Groq TTS terms and testing access
"""

import os
import sys
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv('nAgent/backend/api_keys/.env')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found!")
    sys.exit(1)

print("🔊 Groq TTS Terms Acceptance Guide")
print("=" * 50)

print("\n📋 **Step 1: Accept Terms of Service**")
print("1. Go to: https://console.groq.com/playground")
print("2. Look for 'Text to Speech' or 'TTS' models")
print("3. Find 'playai-tts' model")
print("4. Click 'Accept Terms' or similar button")
print("5. Accept Play.ht's Terms of Service")
print("6. Wait for approval (usually instant)")

print("\n📋 **Step 2: Test TTS Access**")
print("After accepting terms, run this test:")

def test_tts_access():
    """Test if TTS is accessible"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        print("\n🔍 Testing TTS access...")
        
        # Try a simple TTS request
        response = client.audio.speech.create(
            model="playai-tts",
            voice="Mason-PlayAI",
            input="Test",
            response_format="wav"
        )
        
        print("✅ TTS access successful!")
        print("🎉 You can now use TTS in your pipeline!")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ TTS access failed: {error_msg}")
        
        if "terms acceptance" in error_msg.lower():
            print("\n💡 **You still need to accept terms!**")
            print("🔗 Go to: https://console.groq.com/playground")
            print("   Look for TTS models and accept the terms")
        elif "model not found" in error_msg.lower():
            print("\n💡 **Model not available in your account**")
            print("   Contact Groq support or check your plan")
        else:
            print("\n💡 **Unknown error**")
            print("   Check your API key and permissions")
        
        return False

# Test current access
print("\n🧪 **Testing Current TTS Access**")
access_granted = test_tts_access()

if access_granted:
    print("\n🎯 **Next Steps:**")
    print("1. Your TTS is working!")
    print("2. Run the full pipeline: python3 groq_pipeline_final.py")
    print("3. Enjoy speech-to-speech conversations!")
else:
    print("\n🎯 **Next Steps:**")
    print("1. Accept terms at Groq playground")
    print("2. Run this script again to test")
    print("3. Once working, run the full pipeline")

print("\n📚 **Documentation:**")
print("- [Groq TTS Docs](https://console.groq.com/docs/text-to-speech)")
print("- [PlayAI Model Card](https://console.groq.com/docs/model/playai-tts)")
print("- [Groq Playground](https://console.groq.com/playground)")

print("\n💰 **Pricing:**")
print("- $50.00 per million characters")
print("- 20,000 characters per $1")
print("- Very cost-effective for most use cases") 