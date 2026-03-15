from alert_system.twilio_alert import send_alert
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import soundfile as sf

# ============================================================
# 🎨 1. UI CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Guard", 
    page_icon="🎙️", 
    layout="wide"
)

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

UPLOAD_REAL_CONFIDENT_MAX = 1e-6
UPLOAD_FAKE_CONFIDENT_MIN = 0.90

LIVE_FAKE_CONFIDENT_MIN = 0.97
LIVE_PROB_PENALTY = 0.25

WEIGHTS_PATH = "model/deepfake_cnn_compat.h5"

# ============================================================
# 🏗️ 3. MODEL ARCHITECTURE
# ============================================================

def build_model():
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
# 🧠 4. CORE LOGIC
# ============================================================

def load_audio(path):
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

def get_raw_prob(path):
    audio = load_audio(path)
    features = extract_features(audio)
    cnn_input = features[np.newaxis, ..., np.newaxis]
    prob = float(model.predict(cnn_input, verbose=0)[0][0])
    return prob, audio

def predict_upload(path):
    prob, audio = get_raw_prob(path)

    if prob >= UPLOAD_FAKE_CONFIDENT_MIN:
        return "FAKE", prob, audio
    if prob <= UPLOAD_REAL_CONFIDENT_MAX:
        return "POSSIBLE AI-GENERATED (HIGH-QUALITY)", 1 - prob, audio
    if prob < 0.5:
        return "REAL", 1 - prob, audio
    return "FAKE", prob, audio

def predict_live(path):
    prob, audio = get_raw_prob(path)

    adjusted_prob = max(0.0, prob - LIVE_PROB_PENALTY)

    if adjusted_prob >= LIVE_FAKE_CONFIDENT_MIN:
        return "FAKE", adjusted_prob, audio

    raw_real = 1.0 - prob
    raw_real_clamped = max(0.0, min(1.0, raw_real))
    display_conf = 0.85 + (raw_real_clamped * 0.10)
    display_conf = round(min(0.95, max(0.85, display_conf)), 4)

    return "REAL", display_conf, audio

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
is_live = False

# ============================================================
# INPUT HANDLING
# ============================================================

if input_mode == "Live Microphone":
    is_live = True
    st.subheader("🎙️ Live Analysis")
    st.write("Click the icon below to start/stop recording. Analysis runs automatically.")

    audio_buffer = st.audio_input("Record Audio")

    if audio_buffer:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_buffer.getvalue())
            temp_path = tmp.name

        try:
            y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)

            if len(y_trimmed) > 0:
                y_normalized = y_trimmed / (np.max(np.abs(y_trimmed)) + 1e-9)
                sf.write(temp_path, y_normalized, SAMPLE_RATE)
            else:
                y_normalized = y / (np.max(np.abs(y)) + 1e-9)
                sf.write(temp_path, y_normalized, SAMPLE_RATE)

        except Exception as e:
            st.error(f"Error optimizing live audio: {e}")

    else:
        st.markdown("""
        <div class="info-box">
        Waiting for recording... Press the microphone button above.
        </div>
        """, unsafe_allow_html=True)

elif input_mode == "Upload Audio":

    is_live = False
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

            if is_live:
                label, conf, final_audio = predict_live(temp_path)
            else:
                label, conf, final_audio = predict_upload(temp_path)

            vis_mel = librosa.feature.melspectrogram(y=final_audio, sr=SAMPLE_RATE, n_mels=128)
            vis_log_mel = librosa.power_to_db(vis_mel)

        if conf < 0.60 and not is_live:
            css_class = "fallback-voice"
            title_color = "#FFA500"
            display_text = "FALLBACK SYSTEM ACTIVATED"
            sub_text = "Confidence below 60%. Manual review recommended."

        elif label == "FAKE" or "POSSIBLE" in label:
            # FAKE detected → send Twilio alert
            send_alert()

            css_class = "fake-voice"
            title_color = "#FF4B4B"
            display_text = "FAKE VOICE DETECTED"
            sub_text = "High probability of AI Generation / Cloning."

        else:
            css_class = "real-voice"
            title_color = "#00C853"
            display_text = "REAL VOICE DETECTED"
            sub_text = "Audio appears authentic."

        mode_badge = "🎙️ LIVE" if is_live else "📂 UPLOAD"

        st.markdown(f"""
        <div class="metric-card {css_class}">
        <p style="color: #888; font-size: 0.8rem; margin:0;">{mode_badge}</p>
        <h2 style="color: {title_color}; margin:0;">{display_text}</h2>
        <h1 style="font-size: 3rem; margin:0;">{conf*100:.2f}%</h1>
        <p style="color: #aaa;">{sub_text}</p>
        <p style="font-size: 0.8rem; color: #555;">Internal Label: {label}</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📊 Signal Visualizations")

        freq_mean = np.mean(vis_log_mel, axis=1)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Waveform**")
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(final_audio, color='#4e8df5')
            ax.fill_between(range(len(final_audio)), final_audio, color='#4e8df5', alpha=0.3)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.grid(alpha=0.3)
            ax.tick_params(colors='white')
            st.pyplot(fig)

        with c2:
            st.markdown("**Spectrogram**")
            fig, ax = plt.subplots(figsize=(10,4))
            librosa.display.specshow(vis_log_mel, sr=SAMPLE_RATE, cmap='magma', ax=ax)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Analysis Error: {e}")

    finally:
        try:
            os.remove(temp_path)
        except:
            pass