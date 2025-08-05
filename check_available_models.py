#!/usr/bin/env python3
"""
Check available Groq models and their capabilities
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

print("🔍 Checking available Groq models...")

try:
    # Initialize Groq client
    client = Groq(api_key=GROQ_API_KEY)
    
    # Test different model types
    models_to_test = [
        # LLM models
        {"name": "llama3-8b-8192", "type": "LLM"},
        {"name": "llama3.1-8b-8192", "type": "LLM"},
        {"name": "llama3.1-70b-8192", "type": "LLM"},
        {"name": "mixtral-8x7b-32768", "type": "LLM"},
        
        # TTS models (these might not work)
        {"name": "playai-tts", "type": "TTS"},
        {"name": "llama3.2-70b-4096", "type": "TTS"},
        {"name": "llama3-8b-8192", "type": "TTS"},
        
        # Other potential models
        {"name": "gemma2-9b-it", "type": "LLM"},
        {"name": "llama3.1-405b-8192", "type": "LLM"},
    ]
    
    print("\n📋 Testing model availability:")
    print("=" * 50)
    
    working_llm_models = []
    working_tts_models = []
    
    for model_info in models_to_test:
        model_name = model_info["name"]
        model_type = model_info["type"]
        
        print(f"\n🔍 Testing {model_type} model: {model_name}")
        
        try:
            if model_type == "LLM":
                # Test LLM
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                )
                print(f"✅ {model_name} - LLM working")
                working_llm_models.append(model_name)
                
            elif model_type == "TTS":
                # Test TTS
                response = client.audio.speech.create(
                    model=model_name,
                    voice="alloy",
                    input="Test",
                    response_format="wav"
                )
                print(f"✅ {model_name} - TTS working")
                working_tts_models.append(model_name)
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ {model_name} - Error: {error_msg[:100]}...")
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"✅ Working LLM models: {len(working_llm_models)}")
    for model in working_llm_models:
        print(f"   - {model}")
    
    print(f"\n✅ Working TTS models: {len(working_tts_models)}")
    for model in working_tts_models:
        print(f"   - {model}")
    
    if not working_tts_models:
        print("\n⚠️  No TTS models available!")
        print("💡 You may need to:")
        print("   1. Accept terms at https://console.groq.com/playground")
        print("   2. Check Groq documentation for TTS models")
        print("   3. Use a different TTS service (like ElevenLabs)")
    
except Exception as e:
    print(f"❌ Error checking models: {e}")

print("\n📚 Documentation links:")
print("- Groq Models: https://console.groq.com/docs/models")
print("- Groq TTS: https://console.groq.com/docs/audio")
print("- Groq Playground: https://console.groq.com/playground") 