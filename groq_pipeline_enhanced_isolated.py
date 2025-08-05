"""
Enhanced Groq Pipeline (Isolated): Prevents AI from responding to its own speech

This version filters out the AI's own responses and adds better audio isolation.
"""

import asyncio
import json
import os
import sys
import time
import uuid
import re
from typing import Optional, Dict, List, Tuple
from contextlib import suppress
from enum import Enum

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

# Deepgram WebSocket URL - using proven configuration
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

class ConversationType(Enum):
    """Types of conversations the assistant can handle"""
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    EDUCATIONAL = "educational"
    TECHNICAL = "technical"
    CREATIVE = "creative"

class ResponseFilter:
    """Filters out AI's own responses and common echo phrases"""
    
    def __init__(self):
        # Common phrases that the AI might say that could be picked up
        self.ai_response_patterns = [
            r"hello.*great to meet you",
            r"it's great to meet you",
            r"i'm here to help",
            r"what's on your mind",
            r"i'm all ears",
            r"ready to chat",
            r"what would you like to know",
            r"i'm here to listen",
            r"how can i help",
            r"what can i do for you",
            r"tell me more",
            r"that's interesting",
            r"i understand",
            r"that makes sense",
            r"you're right",
            r"absolutely",
            r"definitely",
            r"of course",
            r"certainly",
            r"indeed",
            r"exactly",
            r"precisely",
            r"you're welcome",
            r"my pleasure",
            r"glad to help",
            r"happy to assist",
            r"no problem",
            r"anytime",
            r"sure thing",
            r"absolutely",
            r"definitely",
            r"of course",
            r"certainly",
            r"indeed",
            r"exactly",
            r"precisely",
            r"you're welcome",
            r"my pleasure",
            r"glad to help",
            r"happy to assist",
            r"no problem",
            r"anytime",
            r"sure thing"
        ]
        
        # Compile patterns for faster matching
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.ai_response_patterns]
        
        # Track recent AI responses to avoid echo
        self.recent_responses = []
        self.max_recent_responses = 5
    
    def add_ai_response(self, response: str):
        """Add AI response to recent list for filtering"""
        self.recent_responses.append(response.lower().strip())
        if len(self.recent_responses) > self.max_recent_responses:
            self.recent_responses.pop(0)
    
    def is_ai_echo(self, text: str) -> bool:
        """Check if text is likely an echo of AI's own response"""
        text_lower = text.lower().strip()
        
        # Check against recent AI responses
        for recent_response in self.recent_responses:
            if text_lower in recent_response or recent_response in text_lower:
                return True
        
        # Check against common AI response patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text_lower):
                return True
        
        # Check for very short responses that might be fragments
        if len(text_lower.split()) <= 3 and any(word in text_lower for word in ["hello", "hi", "hey", "you", "me", "i", "am", "is", "are", "the", "and", "or", "but"]):
            return True
        
        return False

class SimpleEmotionDetector:
    """Simple emotion and intent detection"""
    
    def __init__(self):
        self.emotion_keywords = {
            "happy": ["happy", "great", "awesome", "excellent", "wonderful", "amazing", "fantastic", "good", "nice"],
            "sad": ["sad", "upset", "disappointed", "frustrated", "angry", "mad", "terrible", "bad"],
            "curious": ["what", "how", "why", "when", "where", "who", "which", "?"],
            "urgent": ["help", "emergency", "urgent", "quick", "fast", "now", "immediately"],
            "casual": ["hello", "hi", "hey", "how are you", "what's up", "nice to meet you"],
            "technical": ["code", "programming", "algorithm", "function", "debug", "error", "bug", "python", "javascript"],
            "creative": ["story", "write", "create", "imagine", "design", "art", "music", "creative"]
        }
    
    def detect_emotion_and_intent(self, text: str) -> Tuple[str, str]:
        """Detect emotion and intent from text"""
        text_lower = text.lower()
        
        # Detect emotion
        detected_emotion = "neutral"
        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_emotion = emotion
                break
        
        # Detect conversation type
        conversation_type = ConversationType.CASUAL
        if any(word in text_lower for word in self.emotion_keywords["technical"]):
            conversation_type = ConversationType.TECHNICAL
        elif any(word in text_lower for word in self.emotion_keywords["creative"]):
            conversation_type = ConversationType.CREATIVE
        elif any(word in text_lower for word in self.emotion_keywords["curious"]):
            conversation_type = ConversationType.EDUCATIONAL
        
        return detected_emotion, conversation_type.value

class EnhancedConversationContext:
    """Enhanced conversation context with emotion tracking"""
    
    def __init__(self, max_history: int = 10):
        self.conversation_history: List[Dict] = []
        self.max_history = max_history
        self.current_emotion = "neutral"
        self.conversation_type = ConversationType.CASUAL
    
    def add_user_message(self, text: str, emotion: str = "neutral"):
        """Add user message with emotion tracking"""
        self.conversation_history.append({
            "role": "user",
            "content": text,
            "emotion": emotion,
            "timestamp": time.time()
        })
        self.current_emotion = emotion
        self._trim_history()
    
    def add_assistant_message(self, text: str, voice_used: str = "default"):
        """Add assistant response with voice tracking"""
        self.conversation_history.append({
            "role": "assistant", 
            "content": text,
            "voice_used": voice_used,
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

class SimplePromptEngineer:
    """Simple but effective prompt engineering with emotion awareness"""
    
    def __init__(self, context: EnhancedConversationContext):
        self.context = context
        self.emotion_detector = SimpleEmotionDetector()
        
        # Dynamic system prompts based on conversation type
        self.system_prompts = {
            ConversationType.CASUAL: """You are a friendly, empathetic, and engaging conversational AI assistant. 
            You respond naturally and warmly, matching the user's emotional state. Keep responses conversational 
            and personal. Show genuine interest in the user's thoughts and feelings.""",
            
            ConversationType.PROFESSIONAL: """You are a professional, knowledgeable, and efficient AI assistant. 
            You provide clear, concise, and well-structured responses. Maintain a helpful and respectful tone 
            while being informative and solution-oriented.""",
            
            ConversationType.EDUCATIONAL: """You are an educational AI assistant with deep knowledge and teaching skills. 
            You explain complex topics clearly, provide examples, and encourage learning. Be patient, thorough, 
            and adapt your explanations to the user's level of understanding.""",
            
            ConversationType.TECHNICAL: """You are a technical AI assistant with expertise in programming, 
            software development, and technology. You provide accurate, detailed technical guidance, code examples, 
            and troubleshooting help. Be precise and practical in your responses.""",
            
            ConversationType.CREATIVE: """You are a creative AI assistant with imagination and artistic sensibility. 
            You help with creative writing, brainstorming, design ideas, and artistic projects. Be inspiring, 
            original, and supportive of creative expression."""
        }
    
    def build_enhanced_prompt(self, user_input: str) -> List[Dict]:
        """Build an enhanced prompt with emotion and context awareness"""
        
        # Detect emotion and conversation type
        emotion, conv_type = self.emotion_detector.detect_emotion_and_intent(user_input)
        self.context.current_emotion = emotion
        self.context.conversation_type = ConversationType(conv_type)
        
        # Get appropriate system prompt
        system_prompt = self.system_prompts.get(
            self.context.conversation_type, 
            self.system_prompts[ConversationType.CASUAL]
        )
        
        # Add emotion-specific instructions
        emotion_instructions = {
            "happy": "The user seems happy and positive. Match their enthusiasm and energy in your response.",
            "sad": "The user seems upset or frustrated. Be empathetic, supportive, and offer comfort.",
            "curious": "The user is asking questions. Provide clear, helpful, and informative answers.",
            "urgent": "The user needs immediate help. Be direct, efficient, and solution-focused.",
            "casual": "The user is being casual and friendly. Keep the conversation light and engaging.",
            "technical": "The user is asking technical questions. Provide detailed, accurate technical guidance.",
            "creative": "The user is being creative. Be inspiring and supportive of their creative ideas.",
            "neutral": "The user is neutral. Maintain a balanced, helpful tone."
        }
        
        enhanced_system_prompt = f"{system_prompt}\n\n{emotion_instructions.get(emotion, '')}"
        
        # Build messages
        messages = [{"role": "system", "content": enhanced_system_prompt}]
        
        # Add recent conversation context
        recent_context = self.context.get_recent_context()
        messages.extend(recent_context)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages

class SimpleVoiceSelector:
    """Selects appropriate voice based on context and emotion"""
    
    def __init__(self):
        self.voice_configs = {
            "default": {"model": "playai-tts", "voice": "Mason-PlayAI"},
            "friendly": {"model": "playai-tts", "voice": "Fritz-PlayAI"},
            "professional": {"model": "playai-tts", "voice": "Atlas-PlayAI"},
            "warm": {"model": "playai-tts", "voice": "Basil-PlayAI"},
            "clear": {"model": "playai-tts", "voice": "Celeste-PlayAI"},
            "energetic": {"model": "playai-tts", "voice": "Thunder-PlayAI"},
            "calm": {"model": "playai-tts", "voice": "Gail-PlayAI"}
        }
        
        self.emotion_voice_mapping = {
            "happy": "energetic",
            "sad": "calm",
            "curious": "clear",
            "urgent": "professional",
            "casual": "friendly",
            "technical": "professional",
            "creative": "warm",
            "neutral": "default"
        }
    
    def select_voice(self, emotion: str, conversation_type: str) -> Dict:
        """Select voice based on emotion and conversation type"""
        voice_type = self.emotion_voice_mapping.get(emotion, "default")
        return self.voice_configs.get(voice_type, self.voice_configs["default"])

class EnhancedGroqPipeline:
    """Enhanced pipeline with echo prevention and better audio isolation"""
    
    def __init__(self):
        self.context = EnhancedConversationContext()
        self.prompt_engineer = SimplePromptEngineer(self.context)
        self.voice_selector = SimpleVoiceSelector()
        self.response_filter = ResponseFilter()
        self.audio_queue = asyncio.Queue()
        self.is_running = False
        self.emotion_detector = SimpleEmotionDetector()
        self.processing_lock = asyncio.Lock()  # Prevent overlapping processing
        self.is_playing_audio = False  # Track if audio is currently playing
        
    async def audio_generator(self) -> None:
        """Capture audio from microphone and put in queue"""
        def callback(indata, frames, time, status):
            if status:
                print(f"[Audio] {status}", file=sys.stderr)
            # Only add audio to queue if not currently playing AI response
            if not self.is_playing_audio:
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
        """Process text through Groq LLM with enhanced prompting"""
        try:
            # Build enhanced prompt with emotion and context
            messages = self.prompt_engineer.build_enhanced_prompt(text)
            
            # Call Groq LLM with enhanced parameters
            response = groq_llm.chat.completions.create(
                model="llama3-8b-8192",
                messages=messages,
                max_tokens=200,  # Slightly longer for more detailed responses
                temperature=0.8,  # More creative responses
                top_p=0.9,       # Better response quality
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Update conversation context
            emotion, _ = self.emotion_detector.detect_emotion_and_intent(text)
            self.context.add_user_message(text, emotion)
            self.context.add_assistant_message(llm_response)
            
            # Add to response filter
            self.response_filter.add_ai_response(llm_response)
            
            return llm_response
            
        except Exception as e:
            print(f"Error in LLM processing: {e}")
            return "I'm sorry, I encountered an error processing your request."

    async def text_to_speech(self, text: str) -> str:
        """Convert text to speech with intelligent voice selection"""
        try:
            # Generate unique filename
            filename = f"enhanced_response_{uuid.uuid4().hex[:8]}.wav"
            
            # Select appropriate voice based on context
            voice_config = self.voice_selector.select_voice(
                self.context.current_emotion, 
                self.context.conversation_type.value
            )
            
            try:
                # Call Groq TTS with selected voice
                response = groq_tts.audio.speech.create(
                    model=voice_config["model"],
                    voice=voice_config["voice"],
                    input=text,
                    response_format="wav"
                )
                
                # Save audio file
                response.write_to_file(filename)
                print(f"✅ TTS successful with {voice_config['voice']} (emotion: {self.context.current_emotion})")
                
                # Update context with voice used
                self.context.add_assistant_message(text, voice_config["voice"])
                return filename
                
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️  TTS failed with {voice_config['model']}: {error_msg}")
                
                # Fallback to default voice
                fallback_config = self.voice_configs["default"]
                try:
                    response = groq_tts.audio.speech.create(
                        model=fallback_config["model"],
                        voice=fallback_config["voice"],
                        input=text,
                        response_format="wav"
                    )
                    response.write_to_file(filename)
                    print(f"✅ TTS fallback successful with {fallback_config['voice']}")
                    return filename
                except Exception as fallback_error:
                    print(f"❌ TTS fallback also failed: {fallback_error}")
                    return None
            
        except Exception as e:
            print(f"Error in TTS: {e}")
            return None

    async def play_audio(self, filename: str):
        """Play audio file and clean up - with echo prevention"""
        try:
            # Mark that we're playing audio to prevent echo
            self.is_playing_audio = True
            
            # Use system command to play audio (works on macOS) - non-blocking
            os.system(f"afplay {filename} &")  # Run in background
            
            # Wait for audio to finish (approximate)
            await asyncio.sleep(3)  # Give time for audio to finish
            
            # Clean up file
            if os.path.exists(filename):
                os.remove(filename)
                print(f"🗑️  Cleaned up {filename}")
            
            # Mark that we're done playing audio
            self.is_playing_audio = False
                
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.is_playing_audio = False

    async def deepgram_loop(self) -> None:
        """Main Deepgram WebSocket loop with echo prevention"""
        async with connect(
            WS_URL,
            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            ping_interval=5,
            ping_timeout=20,
            close_timeout=0,
        ) as ws:
            print("🎤 Enhanced Listening Mode (Echo Protected)... Press CTRL+C to stop")
            print("🧠 Features: Emotion detection, Context awareness, Dynamic voice selection, Echo prevention")
            print("💡 Speak clearly and wait for the response before speaking again")
            print("🔇 Audio will be muted during AI responses to prevent echo")
            
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
        """Receive and process transcripts with echo prevention"""
        async for message in ws:
            try:
                data = json.loads(message)
                
                # Only process final transcripts
                if not data.get("speech_final", False) and not data.get("is_final", False):
                    continue
                
                transcript = data["channel"]["alternatives"][0].get("transcript", "")
                
                if transcript.strip():
                    # Check if this is likely an echo of AI's own response
                    if self.response_filter.is_ai_echo(transcript):
                        print(f"🔇 Ignoring potential echo: '{transcript}'")
                        continue
                    
                    # Use lock to prevent overlapping processing
                    async with self.processing_lock:
                        # Detect emotion and intent
                        emotion, conv_type = self.emotion_detector.detect_emotion_and_intent(transcript)
                        
                        print(f"\n🎤 You said: {transcript}")
                        print(f"😊 Detected emotion: {emotion}, Conversation type: {conv_type}")
                        
                        # Process through enhanced LLM
                        print("🧠 Processing with enhanced AI...")
                        llm_response = await self.process_with_llm(transcript)
                        print(f"🤖 Assistant: {llm_response}")
                        
                        # Convert to speech with intelligent voice selection
                        print("🔊 Converting to speech with dynamic voice...")
                        audio_file = await self.text_to_speech(llm_response)
                        
                        if audio_file:
                            print("🔊 Playing response...")
                            # Play audio and wait for it to finish
                            await self.play_audio(audio_file)
                        else:
                            print("📝 No audio generated, showing text response only")
                        
                        print("🎤 Ready for next input...")
                    
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
            print("\n👋 Shutting down enhanced pipeline...")
        finally:
            self.is_running = False

async def main():
    """Entry point"""
    pipeline = EnhancedGroqPipeline()
    await pipeline.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exited.") 