#!/usr/bin/env python3
"""
Test script for Groq LLM connection
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

print("🔑 Testing Groq LLM connection...")

try:
    # Initialize Groq client
    client = Groq(api_key=GROQ_API_KEY)
    
    # Test a simple completion
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": "Hello! Can you respond with 'Test successful'?"}
        ],
        max_tokens=50,
        temperature=0.7,
    )
    
    result = response.choices[0].message.content
    print(f"✅ Groq LLM test successful!")
    print(f"🤖 Response: {result}")
    
except Exception as e:
    print(f"❌ Groq LLM test failed: {e}")
    sys.exit(1) 