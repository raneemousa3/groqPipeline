"""
Groq Pipeline (Debug): Enhanced debugging and robust microphone handling

This version includes better debugging and more robust microphone management.
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
load_dotenv(os.path.join(os.path.dirname(__file__), 'nAgent', 'backend', 'api_keys', '.env'))
load_dotenv(os.path.join(os.path.dirname(__file__), 'nAgent', 'backend', '.env'))

# Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not DEEPGRAM_API_KEY or not GROQ_API_KEY:
    raise ValueError("Both DEEPGRAM_API_KEY and GROQ_API_KEY environment variables are required")

# Audio settings
SAMPLE_RATE = 16_000
CHUNK_MS = 100
CHUNK_SIZE = SAMPLE_RATE * CHUNK_MS // 1000

# Deepgram WebSocket URL
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

class PromptEngineer:
    """Handles intelligent prompt generation for the LLM"""
    
    def __init__(self, context: ConversationContext):
        self.context = context
        self.system_prompt = """You are a helpful, friendly, and intelligent assistant. 
        You respond naturally and conversationally. Keep responses concise but informative.
        If the user asks a question, provide a clear and helpful answer.
        If they're just chatting, engage naturally with them."""
    
    def build_prompt(self, user_input: str) -> List[Dict]:
        """Build a complete prompt with context for the LLM"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add recent conversation context
        recent_context = self.context.get_recent_context()
        messages.extend(recent_context)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages

class GroqPipeline:
    """Main pipeline orchestrating STT → LLM → Text Response flow with enhanced debugging"""
    
    def __init__(self):
        self.context = ConversationContext()
        self.prompt_engineer = PromptEngineer(self.context)
        self.audio_queue = asyncio.Queue(maxsize=1000)  # Larger queue
        self.is_running = False
        self.is_ai_speaking = False  # Track when AI is speaking
        self.audio_stream = None
        self.processing_lock = asyncio.Lock()  # Prevent overlapping processing
        self.audio_chunks_sent = 0  # Debug counter
        self.last_activity = time.time()  # Track last activity
        
    async def audio_generator(self) -> None:
        """Capture audio from microphone and put in queue - with enhanced debugging"""
        def callback(indata, frames, time, status):
            if status:
                print(f"[Audio] {status}", file=sys.stderr)
            
            # Only add audio to queue if AI is not currently speaking
            if not self.is_ai_speaking:
                try:
                    self.audio_queue.put_nowait(bytes(indata))
                    self.audio_chunks_sent += 1
                    if self.audio_chunks_sent % 100 == 0:  # Log every 100 chunks
                        print(f"📊 Audio chunks sent: {self.audio_chunks_sent}")
                except asyncio.QueueFull:
                    # If queue is full, skip this chunk
                    print("⚠️  Audio queue full, skipping chunk")
                except Exception as e:
                    print(f"❌ Error in audio callback: {e}")

        try:
            self.audio_stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                dtype="int16",
                channels=1,
                callback=callback,
            )
            self.audio_stream.start()
            print("🎤 Microphone started and listening...")
            
            while self.is_running:
                await asyncio.sleep(0.1)
                # Log activity every 5 seconds
                if time.time() - self.last_activity > 5:
                    print(f"💤 Still listening... (chunks sent: {self.audio_chunks_sent})")
                    self.last_activity = time.time()
                    
        except Exception as e:
            print(f"❌ Error in audio generator: {e}")
        finally:
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
            print("🎤 Microphone stopped.")

    async def process_with_llm(self, text: str) -> str:
        """Process text through Groq LLM and return response"""
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
            return "I'm sorry, I encountered an error processing your request."

    async def text_to_speech(self, text: str) -> str:
        """Convert text to speech using Groq TTS"""
        try:
            # Generate unique filename
            filename = f"response_{uuid.uuid4().hex[:8]}.wav"
            
            # Use correct TTS models from Groq documentation
            tts_configs = [
                {"model": "playai-tts", "voice": "Mason-PlayAI", "description": "English TTS"},
                {"model": "playai-tts", "voice": "Fritz-PlayAI", "description": "English TTS (backup)"},
                {"model": "playai-tts-arabic", "voice": "Ahmad-PlayAI", "description": "Arabic TTS"}
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
                    print(f"✅ TTS successful with {tts_config['description']}: {tts_config['voice']}")
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
        """Play audio file and clean up - with enhanced debugging"""
        try:
            # Mark that AI is speaking to prevent echo
            self.is_ai_speaking = True
            print("🔇 Microphone paused during AI response...")
            
            # Use system command to play audio (works on macOS)
            os.system(f"afplay {filename}")
            
            # Wait a bit after audio finishes to ensure no echo
            await asyncio.sleep(0.5)
            
            # Clean up file after playing
            if os.path.exists(filename):
                os.remove(filename)
                print(f"🗑️  Cleaned up {filename}")
            
            # Mark that AI is done speaking
            self.is_ai_speaking = False
            print("🎤 Microphone active and listening...")
            self.last_activity = time.time()  # Reset activity timer
                
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.is_ai_speaking = False

    async def deepgram_loop(self) -> None:
        """Main Deepgram WebSocket loop for STT"""
        async with connect(
            WS_URL,
            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            ping_interval=5,
            ping_timeout=20,
            close_timeout=0,
        ) as ws:
            print("🎤 Listening (Debug Mode)... Press CTRL+C to stop")
            print("💡 Speak clearly and wait for the response before speaking again")
            print("🔇 Microphone will pause during AI responses")
            print("📊 Debug info will be shown periodically")
            
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
        """Send audio chunks to Deepgram with debugging"""
        chunks_sent = 0
        while self.is_running:
            try:
                chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                await ws.send(chunk)
                chunks_sent += 1
                if chunks_sent % 50 == 0:  # Log every 50 chunks sent to Deepgram
                    print(f"📡 Sent {chunks_sent} chunks to Deepgram")
            except asyncio.TimeoutError:
                # No audio data, continue
                continue
            except Exception as e:
                print(f"❌ Error sending audio: {e}")

    async def receive_transcripts(self, ws) -> None:
        """Receive and process transcripts from Deepgram with debugging"""
        async for message in ws:
            try:
                data = json.loads(message)
                
                # Log all messages for debugging
                if data.get("type") == "Results":
                    print(f"📨 Received message from Deepgram")
                
                # Only process final transcripts
                if not data.get("speech_final", False) and not data.get("is_final", False):
                    continue
                
                transcript = data["channel"]["alternatives"][0].get("transcript", "")
                
                if transcript.strip():
                    # Use lock to prevent overlapping processing
                    async with self.processing_lock:
                        print(f"\n🎤 You said: {transcript}")
                        
                        # Process through LLM
                        print("🤔 Processing...")
                        llm_response = await self.process_with_llm(transcript)
                        print(f"🤖 Assistant: {llm_response}")
                        
                        # Convert to speech
                        print("🔊 Converting to speech...")
                        audio_file = await self.text_to_speech(llm_response)
                        
                        if audio_file:
                            print("🔊 Playing response...")
                            await self.play_audio(audio_file)
                        
                        print("🎤 Ready for next input...")
                        self.last_activity = time.time()  # Reset activity timer
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing transcript: {e}")
                continue
            except Exception as e:
                print(f"❌ Error in receive_transcripts: {e}")

    async def run(self):
        """Main pipeline execution"""
        self.is_running = True
        
        try:
            await asyncio.gather(
                self.audio_generator(),
                self.deepgram_loop()
            )
        except KeyboardInterrupt:
            print("\n👋 Shutting down pipeline...")
        finally:
            self.is_running = False

async def main():
    """Entry point"""
    pipeline = GroqPipeline()
    await pipeline.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exited.") 