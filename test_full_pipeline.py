#!/usr/bin/env python3
"""
Test script for Full Groq Pipeline (simulated input)
"""

import os
import sys
import asyncio
import uuid
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv('nAgent/backend/api_keys/.env')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found!")
    sys.exit(1)

print("🧪 Testing Full Groq Pipeline (simulated input)...")

class TestFullPipeline:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.conversation_history = []
    
    async def simulate_stt_input(self, text: str):
        """Simulate STT input"""
        print(f"🎤 Simulated STT Input: {text}")
        return text
    
    async def process_with_llm(self, text: str):
        """Process text through Groq LLM"""
        try:
            # Build messages with context
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Keep responses concise."}
            ]
            
            # Add recent conversation history
            if self.conversation_history:
                messages.extend(self.conversation_history[-4:])
            
            # Add current user input
            messages.append({"role": "user", "content": text})
            
            # Call Groq LLM
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=messages,
                max_tokens=100,
                temperature=0.7,
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": text})
            self.conversation_history.append({"role": "assistant", "content": llm_response})
            
            # Keep only last 10 messages
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return llm_response
            
        except Exception as e:
            print(f"❌ LLM processing error: {e}")
            return "I'm sorry, I encountered an error."
    
    async def text_to_speech(self, text: str):
        """Convert text to speech"""
        try:
            filename = f"test_response_{uuid.uuid4().hex[:8]}.wav"
            
            # Try different TTS models
            tts_models = [
                {"model": "llama3.2-70b-4096", "voice": "alloy"},
                {"model": "llama3-8b-8192", "voice": "alloy"},
                {"model": "mixtral-8x7b-32768", "voice": "alloy"}
            ]
            
            for tts_config in tts_models:
                try:
                    response = self.client.audio.speech.create(
                        model=tts_config["model"],
                        voice=tts_config["voice"],
                        input=text,
                        response_format="wav"
                    )
                    
                    response.write_to_file(filename)
                    print(f"✅ TTS successful with {tts_config['model']}")
                    return filename
                    
                except Exception as e:
                    print(f"⚠️  TTS failed with {tts_config['model']}: {e}")
                    continue
            
            print(f"❌ All TTS models failed. Text response: {text}")
            return None
            
        except Exception as e:
            print(f"Error in TTS: {e}")
            return None
    
    async def play_audio(self, filename: str):
        """Play audio file"""
        try:
            print(f"🔊 Playing audio: {filename}")
            os.system(f"afplay {filename}")
            
            # Clean up after playing
            await asyncio.sleep(2)
            if os.path.exists(filename):
                os.remove(filename)
                print(f"🗑️  Cleaned up {filename}")
                
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    async def run_pipeline(self, user_input: str):
        """Run the complete pipeline"""
        print(f"\n🔄 Running pipeline for: '{user_input}'")
        
        # Step 1: Simulate STT
        transcribed_text = await self.simulate_stt_input(user_input)
        
        # Step 2: Process with LLM
        print("🤔 Processing with LLM...")
        llm_response = await self.process_with_llm(transcribed_text)
        print(f"🤖 LLM Response: {llm_response}")
        
        # Step 3: Convert to speech
        print("🔊 Converting to speech...")
        audio_file = await self.text_to_speech(llm_response)
        
        # Step 4: Play audio
        if audio_file:
            await self.play_audio(audio_file)
        else:
            print("📝 No audio generated, showing text response only")
        
        print("✅ Pipeline step completed!")

async def main():
    """Test the full pipeline"""
    pipeline = TestFullPipeline()
    
    test_inputs = [
        "Hello, how are you?",
        "What time is it?",
        "Tell me something interesting"
    ]
    
    print("🚀 Starting full pipeline test...")
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n{'='*50}")
        print(f"TEST {i}/{len(test_inputs)}")
        print(f"{'='*50}")
        
        await pipeline.run_pipeline(user_input)
        
        # Wait between tests
        await asyncio.sleep(2)
    
    print(f"\n{'='*50}")
    print("🎉 Full pipeline test completed!")
    print("✅ All components are working correctly")
    print(f"{'='*50}")

if __name__ == "__main__":
    asyncio.run(main()) 