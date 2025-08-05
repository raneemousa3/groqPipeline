#!/usr/bin/env python3
"""
Setup script for Groq Pipeline
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def check_env_file():
    """Check if .env file exists"""
    if not os.path.exists(".env"):
        print("⚠️  .env file not found!")
        print("📝 Please create a .env file with your API keys:")
        print("   DEEPGRAM_API_KEY=your_deepgram_api_key_here")
        print("   GROQ_API_KEY=your_groq_api_key_here")
        return False
    else:
        print("✅ .env file found!")
        return True

def main():
    """Main setup function"""
    print("🚀 Setting up Groq Pipeline...")
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check environment file
    check_env_file()
    
    print("\n🎉 Setup complete!")
    print("📖 To run the pipeline:")
    print("   python groq_pipeline.py")
    print("\n🔑 Make sure to add your API keys to the .env file!")

if __name__ == "__main__":
    main() 