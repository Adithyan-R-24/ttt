import os
import time
import base64
import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from gtts import gTTS
import google.generativeai as genai
import re
import tempfile # Added for temporary file handling

# ==== Configuration ====
st.set_page_config(page_title="🗣 தமிழ் உதவியாளர்", layout="wide")

# ==== Gemini Setup ====
# Replace with your actual API key
# SECURITY WARNING: Hardcoding API keys directly in source code is NOT recommended for production applications.
# Consider using environment variables or Streamlit Secrets for better security.
API_KEY = "AIzaSyC9mr7-JTjX6ZlHVfGzxRZ1StM2QCBIKCg" # Replace with your actual API key
genai.configure(api_key=API_KEY)

try:
    model = genai.GenerativeModel("models/gemini-1.5-flash")
except Exception as e:
    st.error(f"❌ Gemini Model Initialization Error: {e}")
    st.stop() # Stop execution if model fails to load

# ==== Chat Bubble Style ====
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        width: 100%;
        max-width: 90%; /* Increased max-width */
        margin: 20px auto; /* Added margin for better spacing */
        padding-bottom: 150px; /* Add padding to make space for input area */
    }
    .chat-bubble {
        max-width: 75%;
        padding: 10px 15px;
        margin: 5px 0; /* Reduced margin */
        border-radius: 15px;
        line-height: 1.6;
        font-size: 16px;
        clear: both; /* Ensure proper floating */
        word-wrap: break-word; /* Ensure long words break */
    }
    .user-bubble {
        background-color: #38bdf8;
        color: black;
        float: right;
        text-align: right;
        margin-left: 25%; /* Auto margin pushes it right */
    }
    .bot-bubble {
        background-color: #14532d; /* Dark green */
        color: white;
        float: left;
        text-align: left;
        margin-right: 25%; /* Auto margin pushes it left */
    }
    .input-area-container {
        position: fixed; /* Fixed position */
        bottom: 0; /* At the bottom */
        left: 0; /* Spans full width */
        width: 100%;
        background-color: #1e1e1e; /* Dark background for input area */
        padding: 10px 0;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1); /* Shadow at the top */
        z-index: 1000; /* Ensure it's on top */
    }
    .input-area {
        display: flex;
        justify-content: center; /* Center the button */
        width: 100%;
        max-width: 90%;
        margin: 0 auto;
    }
    .audio-button {
        padding: 12px 20px;
        border-radius: 25px;
        background-color: #38bdf8;
        color: white;
        font-size: 18px;
        cursor: pointer;
        border: none;
        display: inline-block; /* Ensure button properties */
    }
    .audio-button:hover {
        background-color: #0ea5e9;
    }
     .refresh-button {
        position: fixed;
        top: 10px;
        right: 10px;
        padding: 10px 15px;
        border-radius: 5px;
        background-color: #f97316;
        color: white;
        font-size: 14px;
        cursor: pointer;
        border: none;
        z-index: 1000; /* Ensure it's on top */
    }
    .refresh-button:hover {
        background-color: #fb923c;
    }
    .audio-player-container {
        margin-top: 10px;
        text-align: left; /* Align audio player to the left */
    }
    </style>
""", unsafe_allow_html=True)

# ==== Helper: Record Audio ====
def record_audio(duration=8, samplerate=16000):
    """Records audio for a given duration and returns the audio data."""
    st.info("🎤 தெளிவாகவும், அமைதியான சூழலிலும் பேசுங்கள்... பதிவு தொடங்குகிறது...")
    time.sleep(0.5)
    try:
        # Check available devices
        devices = sd.query_devices()
        # print("Available audio devices:")
        # for i, device in enumerate(devices):
        #     print(f"{i}: {device['name']}")

        # You might need to select a specific device index if default doesn't work
        # sd.default.device = YOUR_DEVICE_INDEX # Uncomment and set your device index if needed

        recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()  # Wait until recording is finished
        st.info("🎤 பதிவு முடிந்தது!")
        time.sleep(0.2)
        return recording, samplerate
    except Exception as e:
        st.error(f"❌ Audio Recording Error: {e}")
        st.warning("உங்கள் மைக்ரோஃபோன் சரியாக இணைக்கப்பட்டுள்ளதா மற்றும் அனுமதி வழங்கப்பட்டுள்ளதா என்பதைச் சரிபார்க்கவும்.") # Check if mic is connected and permissions granted
        return None, None


# ==== Helper: Save Audio to WAV ====
def save_audio_to_wav(audio_data, samplerate):
    """Saves numpy audio data to a temporary WAV file."""
    if audio_data is None or samplerate is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            filepath = tmp_file.name
            wav.write(filepath, samplerate, audio_data)
            return filepath
    except Exception as e:
        st.error(f"❌ Error saving WAV file: {e}")
        return None

# ==== Helper: Audio to Base64 ====
def audio_to_base64(filepath):
    """Converts an audio file to a Base64 string."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        st.error(f"❌ Error encoding audio to Base64: {e}")
        return None

# ==== Gemini Transcription ====
def transcribe_with_gemini(audio_path):
    """Transcribes audio using the Gemini model."""
    if audio_path is None or not os.path.exists(audio_path):
        return "⚠ ஆடியோ கோப்பு கிடைக்கவில்லை." # Audio file not available

    audio_base64 = audio_to_base64(audio_path)
    if audio_base64 is None:
        return "⚠ ஆடியோவை செயலாக்க முடியவில்லை." # Could not process audio

    prompt = (
        "இந்த ஆடியோவில் பேசப்படுவதைக் கவனமாகக் கேட்டு, "
        "அதை தமிழ் எழுத்தாக மாற்றவும். வேறு எந்த விளக்கமும் சேர்க்க வேண்டாம்." # Added instruction to not add other explanations
    )
    try:
        response = model.generate_content(
            [prompt, {"mime_type": "audio/wav", "data": audio_base64}]
        )
        # Clean up potential markdown artifacts from the response
        text = response.text.strip()
        text = text.replace("json", "").replace("", "").strip() # Remove potential markdown
        text = re.sub(r"^\s*Transcription:\s*", "", text, flags=re.IGNORECASE) # Remove potential "Transcription:" prefix
        return text if text else "⚠ டிரான்ஸ்கிரிப்ஷன் எதுவும் கிடைக்கவில்லை." # No transcription available

    except Exception as e:
        st.error(f"❌ Gemini Speech Error: {e}")
        return f"⚠ டிரான்ஸ்கிரிப்ஷன் பிழை: {e}" # Transcription error

# ==== Gemini Chat ====
def query_gemini_tamil(prompt_text):
    """Queries the Gemini model with chat history context."""
    if not prompt_text:
        return "நீங்கள் என்ன பேசினீர்கள் என்று எனக்குப் புரியவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்." # I didn't understand what you said. Please try again.

    context = ""
    # Include the last 6 turns of the conversation in the prompt
    if st.session_state.chat_history:
        context += "முந்தைய உரையாடல்:\n"
        # Iterate through tuples (role, text)
        for i, (role, text) in enumerate(st.session_state.chat_history[-12:]): # Last 12 entries for 6 turns
            if role == "user":
                context += f"பயனர்: {text}\n"
            else:
                context += f"உதவியாளர்: {text}\n"
        context += "\n"

    full_prompt = (
        f"{context}இப்போது பயனர் கேட்கும் கேள்வி: {prompt_text}\n\n"
        "இந்தக் கேள்விக்கு இயற்கையான முறையில் தமிழில் பதிலளிக்கவும். வேறு எந்த மொழியிலும் பதிலளிக்க வேண்டாம். பதிலில் மார்க் டவுன் குறியீடுகளை (Markdown) தவிர்க்கவும்." # Respond naturally in Tamil, avoid other languages and markdown
    )

    try:
        response = model.generate_content(full_prompt)
        # Clean up potential markdown artifacts from the response
        text = response.text.strip()
        text = text.replace("```", "").strip() # Remove potential markdown code blocks
        return text if text else "மன்னிக்கவும், என்னால் பதிலளிக்க முடியவில்லை." # Sorry, I could not respond.

    except Exception as e:
        st.error(f"❌ Gemini Chat Error: {e}")
        return f"⚠ பதிலளிப்பதில் பிழை: {e}" # Error in responding

# ==== Speak Tamil ====
def speak_tamil(text):
    """Converts text to speech using gTTS and returns the audio file path."""
    if not text or text.startswith("⚠"): # Don't try to speak error messages
         return None

    # Clean unwanted characters and punctuation from the response
    # Keep Tamil letters, numbers, basic punctuation, and spaces
    cleaned_text = re.sub(r"[^\w\s.,?!()\u0B80-\u0BFF]", "", text)
    cleaned_text = cleaned_text.replace("-", " to ").replace("—", " to ") # Convert ranges to words (with spaces)
    cleaned_text = re.sub(r"(\d+)\s?(\d+)", r"\1 to \2", cleaned_text) # Range like 3 5 -> 3 to 5

    if not cleaned_text.strip():
        return None # Don't generate audio for empty string after cleaning

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            filename = tmp_file.name
            tts = gTTS(text=cleaned_text, lang='ta')
            tts.save(filename)
            return filename
    except Exception as e:
        st.error(f"❌ Text-to-Speech Error: {e}")
        return None

def render_audio_player(filepath):
    """Renders an HTML audio player for a given file."""
    if filepath is None or not os.path.exists(filepath):
        return

    audio_base64 = audio_to_base64(filepath)
    if audio_base64:
        audio_html = f"""
            <div class="audio-player-container">
                <audio autoplay controls>
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            </div>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        # Clean up the temporary audio file after rendering
        # Note: This might sometimes delete before playback finishes, depending on browser caching.
        # For robustness in production, proper temp file management or client-side cleanup is better.
        try:
            os.unlink(filepath)
        except OSError as e:
             print(f"Error removing temp audio file {filepath}: {e}")


# ==== Session Memory ====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Limit to last 6 conversations (12 entries)
if len(st.session_state.chat_history) > 12:
    st.session_state.chat_history = st.session_state.chat_history[-12:]

# ==== Clear chat history on refresh button click ====
if st.button("🔄 Refresh", key="refresh_button_top", help="Click to refresh the page and start a new conversation."):
     st.session_state.chat_history = []
     st.session_state.last_audio = None # Also clear last audio
     st.rerun() # Rerun to clear the display


# ==== Chat UI ====
st.title("🗣 தமிழ் உதவியாளர் (Gemini Flash 1.5)")

# Refresh button (redundant with the one above, keeping one)
# st.markdown('<div class="refresh-button"><button onclick="window.location.reload();">🔄 Refresh</button></div>', unsafe_allow_html=True)


# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
# Display history
for role, text in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="chat-bubble user-bubble">{text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble bot-bubble">{text}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input area (fixed at the bottom)
st.markdown('<div class="input-area-container"><div class="input-area">', unsafe_allow_html=True)
if st.button("🎙 பதிவு செய் (8 விநாடிகள்)", key="record_button", use_container_width=False):
    audio_data, samplerate = record_audio()
    if audio_data is not None and samplerate is not None:
        # Save to temporary file
        audio_path = save_audio_to_wav(audio_data, samplerate)

        if audio_path:
            st.success("✅ பதிவு வெற்றிகரமாக சேமிக்கப்பட்டது. செயல்படுத்துகிறேன்...") # Recording successfully saved

            # Transcription
            user_text = transcribe_with_gemini(audio_path)

            # Clean up temporary recording file immediately after transcription attempt
            try:
                os.unlink(audio_path)
            except OSError as e:
                 print(f"Error removing temp recording file {audio_path}: {e}")

            st.session_state.chat_history.append(("user", user_text))

            if not user_text.startswith("⚠"): # Only query Gemini if transcription was not an error
                # Gemini Response
                bot_text = query_gemini_tamil(user_text)
                st.session_state.chat_history.append(("bot", bot_text))

                # Speak and store audio
                audio_file = speak_tamil(bot_text)
                st.session_state.last_audio = audio_file # Store path to temporary audio file
            else:
                 # If transcription failed, add a default bot message
                 st.session_state.chat_history.append(("bot", "மன்னிக்கவும், உங்கள் பேச்சைப் புரிந்துகொள்ள முடியவில்லை.")) # Sorry, could not understand your speech.
                 st.session_state.last_audio = None # No audio to play


            st.rerun() # Rerun to update the chat history and potentially the audio player
        else:
            st.error("❌ ஆடியோ கோப்பை உருவாக்க முடியவில்லை.") # Could not create audio file

st.markdown('</div></div>', unsafe_allow_html=True)


# Show last audio output for replay below the chat messages
# This needs to be outside the fixed input area container
if "last_audio" in st.session_state and st.session_state.last_audio:
     # Add a placeholder below the chat container for the audio player
     # Use a unique key or position carefully to avoid disappearing
     st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True) # Add some space
     render_audio_player(st.session_state.last_audio)
     # Note: Temporary audio file cleanup is attempted within render_audio_player