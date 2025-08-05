"""
Groq Pipeline: Arabic Speech → Text → LLM Processing → Arabic Speech Response

This pipeline integrates:
1. Real-time STT (Deepgram) - supports Arabic
2. LLM processing (Groq) - can handle Arabic text
3. Arabic TTS generation (Groq PlayAI Arabic)
4. Context management for Arabic conversations
"""

import asyncio
import json
import os
import sys
import time
import uuid
from typing import Optional, Dict, List
from contextlib import suppress

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'nAgent', 'backend'))

import sounddevice as sd
from websockets.legacy.client import connect
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from multiple possible locations
load_dotenv()  # Try current directory
load_dotenv(os.path.join(os.path.dirname(__file__), 'nAgent', 'backend', 'api_keys', '.env'))  # Try api_keys folder
load_dotenv(os.path.join(os.path.dirname(__file__), 'nAgent', 'backend', '.env'))  # Try backend folder

# Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not DEEPGRAM_API_KEY or not GROQ_API_KEY:
    raise ValueError("Both DEEPGRAM_API_KEY and GROQ_API_KEY environment variables are required")

# Audio settings
SAMPLE_RATE = 16_000
CHUNK_MS = 100
CHUNK_SIZE = SAMPLE_RATE * CHUNK_MS // 1000

# Deepgram WebSocket URL (will auto-detect Arabic)
WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&punctuate=true"
    "&interim_results=true"
)

# Groq clients
groq_llm = Groq(api_key=GROQ_API_KEY)
groq_tts = Groq(api_key=GROQ_API_KEY)

class ConversationContext:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.conversation_history: List[Dict] = []
        self.max_history = max_history
    
    def add_user_message(self, text: str):
        """Add user message to conversation history"""
        self.conversation_history.append({
            "role": "user",
            "content": text,
            "timestamp": time.time()
        })
        self._trim_history()
    
    def add_assistant_message(self, text: str):
        """Add assistant response to conversation history"""
        self.conversation_history.append({
            "role": "assistant", 
            "content": text,
            "timestamp": time.time()
        })
        self._trim_history()
    
    def get_recent_context(self, num_messages: int = 5) -> List[Dict]:
        """Get recent conversation context for LLM"""
        recent = self.conversation_history[-num_messages:] if self.conversation_history else []
        return [{"role": msg["role"], "content": msg["content"]} for msg in recent]
    
    def _trim_history(self):
        """Keep only the most recent messages"""
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

class ArabicPromptEngineer:
    """Handles intelligent prompt generation for Arabic conversations"""
    
    def __init__(self, context: ConversationContext):
        self.context = context
        self.system_prompt = """أنت مساعد ذكي ومفيد تتحدث باللغة العربية. 
        تجيب بشكل طبيعي ومحادثة. احتفظ بالإجابات موجزة ولكن مفيدة.
        إذا سأل المستخدم سؤالاً، قدم إجابة واضحة ومفيدة.
        إذا كانوا يتحدثون فقط، تفاعل معهم بشكل طبيعي."""
    
    def build_prompt(self, user_input: str) -> List[Dict]:
        """Build a complete prompt with context for the LLM"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add recent conversation context
        recent_context = self.context.get_recent_context()
        messages.extend(recent_context)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages

class ArabicGroqPipeline:
    """Main pipeline orchestrating Arabic STT → LLM → Arabic TTS flow"""
    
    def __init__(self):
        self.context = ConversationContext()
        self.prompt_engineer = ArabicPromptEngineer(self.context)
        self.audio_queue = asyncio.Queue()
        self.is_running = False
        
    async def audio_generator(self) -> None:
        """Capture audio from microphone and put in queue"""
        def callback(indata, frames, time, status):
            if status:
                print(f"[Audio] {status}", file=sys.stderr)
            self.audio_queue.put_nowait(bytes(indata))

        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            dtype="int16",
            channels=1,
            callback=callback,
        )
        stream.start()
        try:
            while self.is_running:
                await asyncio.sleep(0.1)
        finally:
            stream.stop()
            stream.close()

    async def process_with_llm(self, text: str) -> str:
        """Process Arabic text through Groq LLM and return response"""
        try:
            # Build intelligent prompt with context
            messages = self.prompt_engineer.build_prompt(text)
            
            # Call Groq LLM
            response = groq_llm.chat.completions.create(
                model="llama3-8b-8192",  # Fast and efficient model
                messages=messages,
                max_tokens=150,  # Keep responses concise
                temperature=0.7,  # Balanced creativity
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Update conversation context
            self.context.add_user_message(text)
            self.context.add_assistant_message(llm_response)
            
            return llm_response
            
        except Exception as e:
            print(f"Error in LLM processing: {e}")
            return "عذراً، واجهت خطأ في معالجة طلبك."

    async def text_to_speech(self, text: str) -> str:
        """Convert Arabic text to speech using Groq Arabic TTS"""
        try:
            # Generate unique filename
            filename = f"arabic_response_{uuid.uuid4().hex[:8]}.wav"
            
            # Use Arabic TTS models
            tts_configs = [
                {"model": "playai-tts-arabic", "voice": "Ahmad-PlayAI", "description": "Arabic TTS - Ahmad"},
                {"model": "playai-tts-arabic", "voice": "Amira-PlayAI", "description": "Arabic TTS - Amira"},
                {"model": "playai-tts-arabic", "voice": "Khalid-PlayAI", "description": "Arabic TTS - Khalid"},
                {"model": "playai-tts-arabic", "voice": "Nasser-PlayAI", "description": "Arabic TTS - Nasser"},
                # Fallback to English TTS if Arabic fails
                {"model": "playai-tts", "voice": "Mason-PlayAI", "description": "English TTS (fallback)"}
            ]
            
            for tts_config in tts_configs:
                try:
                    # Call Groq TTS with correct parameters
                    response = groq_tts.audio.speech.create(
                        model=tts_config["model"],
                        voice=tts_config["voice"],
                        input=text,
                        response_format="wav"
                    )
                    
                    # Save audio file
                    response.write_to_file(filename)
                    print(f"✅ TTS successful with {tts_config['description']}")
                    return filename
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"⚠️  TTS failed with {tts_config['model']}: {error_msg}")
                    continue
            
            # If all TTS models fail, just print the text
            print(f"❌ All TTS models failed. Text response: {text}")
            return None
            
        except Exception as e:
            print(f"Error in TTS: {e}")
            return None

    async def play_audio(self, filename: str):
        """Play audio file and clean up"""
        try:
            # Use system command to play audio (works on macOS)
            os.system(f"afplay {filename}")
            
            # Clean up file after playing
            await asyncio.sleep(0.5)  # Give time for audio to finish
            if os.path.exists(filename):
                os.remove(filename)
                print(f"🗑️  Cleaned up {filename}")
                
        except Exception as e:
            print(f"Error playing audio: {e}")

    async def deepgram_loop(self) -> None:
        """Main Deepgram WebSocket loop for Arabic STT"""
        async with connect(
            WS_URL,
            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            ping_interval=5,
            ping_timeout=20,
            close_timeout=0,
        ) as ws:
            print("🎤 الاستماع... اضغط CTRL+C للإيقاف")
            print("💬 الردود ستظهر كنص (يمكن إضافة TTS لاحقاً)")
            
            send_task = asyncio.create_task(self.send_audio(ws))
            receive_task = asyncio.create_task(self.receive_transcripts(ws))

            done, pending = await asyncio.wait(
                {send_task, receive_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def send_audio(self, ws) -> None:
        """Send audio chunks to Deepgram"""
        while self.is_running:
            chunk = await self.audio_queue.get()
            await ws.send(chunk)

    async def receive_transcripts(self, ws) -> None:
        """Receive and process Arabic transcripts from Deepgram"""
        async for message in ws:
            try:
                data = json.loads(message)
                
                # Only process final transcripts
                if not data.get("speech_final", False) and not data.get("is_final", False):
                    continue
                
                transcript = data["channel"]["alternatives"][0].get("transcript", "")
                
                if transcript.strip():
                    print(f"\n🎤 قلت: {transcript}")
                    
                    # Process through LLM
                    print("🤔 معالجة...")
                    llm_response = await self.process_with_llm(transcript)
                    print(f"🤖 المساعد: {llm_response}")
                    
                    # Convert to speech
                    print("🔊 تحويل إلى كلام...")
                    audio_file = await self.text_to_speech(llm_response)
                    
                    if audio_file:
                        print("🔊 تشغيل الرد...")
                        await self.play_audio(audio_file)
                    
                    print("🎤 جاهز للإدخال التالي...")
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing transcript: {e}")
                continue

    async def run(self):
        """Main pipeline execution"""
        self.is_running = True
        
        try:
            await asyncio.gather(
                self.audio_generator(),
                self.deepgram_loop()
            )
        except KeyboardInterrupt:
            print("\n👋 إيقاف التشغيل...")
        finally:
            self.is_running = False

async def main():
    """Entry point"""
    pipeline = ArabicGroqPipeline()
    await pipeline.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exited.") 