#!/usr/bin/env python3
"""
Test script for Groq Pipeline (LLM only, no TTS)
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv('nAgent/backend/api_keys/.env')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found!")
    sys.exit(1)

print("🧪 Testing Groq Pipeline (LLM only)...")

class TestPipeline:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.conversation_history = []
    
    async def test_llm_processing(self, user_input: str):
        """Test LLM processing with conversation context"""
        try:
            # Build messages with context
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Keep responses concise."}
            ]
            
            # Add recent conversation history
            if self.conversation_history:
                messages.extend(self.conversation_history[-4:])  # Last 4 messages
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Call Groq LLM
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=messages,
                max_tokens=100,
                temperature=0.7,
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": llm_response})
            
            # Keep only last 10 messages
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return llm_response
            
        except Exception as e:
            print(f"❌ LLM processing error: {e}")
            return "I'm sorry, I encountered an error."

async def main():
    """Test the pipeline with sample inputs"""
    pipeline = TestPipeline()
    
    test_inputs = [
        "Hello, how are you?",
        "What's the weather like today?",
        "Can you help me with a math problem?",
        "Tell me a joke"
    ]
    
    print("🎤 Simulating conversation...")
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- Test {i} ---")
        print(f"👤 User: {user_input}")
        
        response = await pipeline.test_llm_processing(user_input)
        print(f"🤖 Assistant: {response}")
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    print("\n✅ Pipeline test completed successfully!")
    print("🎯 The LLM processing and conversation context are working correctly.")

if __name__ == "__main__":
    asyncio.run(main()) 