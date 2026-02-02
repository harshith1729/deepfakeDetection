import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import soundfile as sf # Required for saving the cleaned audio

# ============================================================
# 🎨 1. UI CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Guard", 
    page_icon="🎙️", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .metric-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .real-voice { border-left: 8px solid #00C853; }
    .fake-voice { border-left: 8px solid #FF4B4B; }
    .suspicious-voice { border-left: 8px solid #FFD700; }
    .fallback-voice { border-left: 8px solid #FFA500; }
    .stAudioInput > label { display: none; }
    .info-box {
        background-color: #1e252b;
        color: #4e8df5; 
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #4e8df5;
        font-family: monospace;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ⚙️ 2. CONSTANTS
# ============================================================

SAMPLE_RATE = 16000
DURATION = 4
SAMPLES = SAMPLE_RATE * DURATION
TIME_FRAMES = 126
FEATURE_DIM = 154

REAL_CONFIDENT_MAX = 1e-6      
FAKE_CONFIDENT_MIN = 0.90      
WEIGHTS_PATH = "model/deepfake_cnn_compat.h5"

# ============================================================
# 🏗️ 3. MODEL ARCHITECTURE
# ============================================================

def build_model():
    model = models.Sequential([
        layers.Input(shape=(FEATURE_DIM, TIME_FRAMES, 1)),
        
        # Block 1
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Block 2
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Block 3
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Block 4 -- REMOVED to match the 8-layer weights file
        # layers.Conv2D(128, (3,3), activation='relu'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

    model = models.Sequential([
        layers.Input(shape=(FEATURE_DIM, TIME_FRAMES, 1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

@st.cache_resource
def load_system():
    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"❌ Critical Error: Weights file not found at {WEIGHTS_PATH}")
        st.stop()
    try:
        model = build_model()
        model.load_weights(WEIGHTS_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model weights: {e}")
        st.stop()

model = load_system()

# ============================================================
# 🧠 4. CORE LOGIC (UNTOUCHED FOR UPLOADS)
# ============================================================

def load_audio(path):
    # Standard logic for file uploads
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    return audio

def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128)
    log_mel = librosa.power_to_db(mel)

    min_frames = min(mfcc.shape[1], delta.shape[1], log_mel.shape[1])

    features = np.vstack([
        mfcc[:, :min_frames],
        delta[:, :min_frames],
        log_mel[:, :min_frames]
    ])

    features = (features - features.mean()) / (features.std() + 1e-6)

    if features.shape[1] > TIME_FRAMES:
        features = features[:, :TIME_FRAMES]
    else:
        features = np.pad(features, ((0,0),(0,TIME_FRAMES - features.shape[1])), mode="constant")

    return features.astype(np.float32)

def predict_audio_logic(path):
    audio = load_audio(path)
    features = extract_features(audio)

    cnn_input = features[np.newaxis, ..., np.newaxis]
    prob = float(model.predict(cnn_input, verbose=0)[0][0])

    if prob >= FAKE_CONFIDENT_MIN: return "FAKE", prob, audio
    if prob <= REAL_CONFIDENT_MAX: return "POSSIBLE AI-GENERATED (HIGH-QUALITY)", 1 - prob, audio
    if prob < 0.5: return "REAL", 1 - prob, audio
    return "FAKE", prob, audio

# ============================================================
# 🖥️ 5. MAIN UI LAYOUT
# ============================================================

st.markdown("# 🎙️ Deepfake Guard")
st.markdown("### Advanced Voice Authentication & Anti-Spoofing")
st.markdown("---")

with st.sidebar:
    st.header("Input Configuration")
    input_mode = st.radio("Choose Input Type:", ["Live Microphone", "Upload Audio"])
    st.markdown("---")
    st.info("ℹ️ System analyzes ~4 seconds of audio.")

temp_path = None

# ============================================================
# 🚨 CRITICAL FIX FOR LIVE MIC ONLY
# ============================================================
if input_mode == "Live Microphone":
    st.subheader("🎙️ Live Analysis")
    st.write("Click the icon below to start/stop recording. The analysis will run automatically.")
    
    audio_buffer = st.audio_input("Record Audio")
    
    if audio_buffer:
        # 1. Save raw recording to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_buffer.getvalue())
            temp_path = tmp.name
            
        # 2. PRE-PROCESS (Fixes "Always Fake" for Live Audio)
        # We clean the audio file BEFORE sending it to the untouched prediction logic.
        try:
            # Load raw file
            y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
            
            # Trim silence
            # top_db=30 trims anything quieter than 30dB below peak (removes silence/hiss)
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            
            # Normalize Volume
            # This makes your voice 'loud' enough for the model to see it clearly
            if len(y_trimmed) > 0:
                y_trimmed = y_trimmed / np.max(np.abs(y_trimmed))
                
                # Save the CLEANED audio back to temp_path
                sf.write(temp_path, y_trimmed, SAMPLE_RATE)
            
        except Exception as e:
            st.error(f"Error optimizing live audio: {e}")

    else:
        st.markdown("""
            <div class="info-box">
                Waiting for recording... Press the microphone button above.
            </div>
        """, unsafe_allow_html=True)

elif input_mode == "Upload Audio":
    st.subheader("📂 File Analysis")
    uploaded = st.file_uploader("Drop audio file here (WAV/MP3)", type=["wav", "mp3"])
    
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            temp_path = tmp.name
    else:
         st.markdown("""
            <div class="info-box">
                Waiting for file upload...
            </div>
        """, unsafe_allow_html=True)

# ============================================================
# 🔍 6. ANALYSIS & RESULTS
# ============================================================

if temp_path is not None:
    st.divider()
    
    try:
        with st.spinner("Processing audio signature..."):
            # We call the EXACT same logic function. 
            # For live mic, 'temp_path' now contains the cleaned/boosted audio.
            label, conf, final_audio = predict_audio_logic(temp_path)
            
            # Visualization calculations
            vis_mel = librosa.feature.melspectrogram(y=final_audio, sr=SAMPLE_RATE, n_mels=128)
            vis_log_mel = librosa.power_to_db(vis_mel)

        # Result Logic
        if conf < 0.60:
            css_class = "fallback-voice"
            title_color = "#FFA500" 
            display_text = "FALLBACK SYSTEM ACTIVATED"
            sub_text = "Confidence below 60%. Manual review recommended."
        elif label == "REAL":
            css_class = "real-voice"
            title_color = "#00C853"
            display_text = "REAL VOICE DETECTED"
            sub_text = "Audio appears authentic."
        elif "POSSIBLE" in label:
            css_class = "suspicious-voice"
            title_color = "#FFD700"
            display_text = "SUSPICIOUS (HIGH QUALITY)"
            sub_text = "Model detected anomalies typical of high-quality cloning."
        else:
            css_class = "fake-voice"
            title_color = "#FF4B4B"
            display_text = "FAKE VOICE DETECTED"
            sub_text = "High probability of AI Generation / Cloning."

        st.markdown(f"""
            <div class="metric-card {css_class}">
                <h2 style="color: {title_color}; margin:0;">{display_text}</h2>
                <h1 style="font-size: 3rem; margin:0;">{conf*100:.2f}%</h1>
                <p style="color: #aaa;">{sub_text}</p>
                <p style="font-size: 0.8rem; color: #555;">Internal Label: {label}</p>
            </div>
        """, unsafe_allow_html=True)

        # Visualizations
        st.subheader("📊 Signal Visualizations")
        freq_mean = np.mean(vis_log_mel, axis=1)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Waveform**")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(final_audio, color='#4e8df5', linewidth=1)
            ax.fill_between(range(len(final_audio)), final_audio, color='#4e8df5', alpha=0.3)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.grid(alpha=0.3)
            ax.tick_params(colors='white')
            st.pyplot(fig)

        with c2:
            st.markdown("**Spectrogram**")
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.specshow(vis_log_mel, sr=SAMPLE_RATE, cmap='magma', ax=ax)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            st.pyplot(fig)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Frequency Line**")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(freq_mean, color='#00d4ff')
            ax.fill_between(range(len(freq_mean)), freq_mean, color='#00d4ff', alpha=0.2)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.grid(alpha=0.3)
            ax.tick_params(colors='white')
            st.pyplot(fig)

        with c4:
            st.markdown("**Energy Bar**")
            fig, ax = plt.subplots(figsize=(10, 4))
            bins = np.array_split(freq_mean, 15)
            means = [np.mean(b) for b in bins]
            ax.bar(range(15), means, color='#d63384', alpha=0.8)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Analysis Error: {e}")
    finally:
        try: os.remove(temp_path)
        except: pass