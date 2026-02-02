import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ============================================================
# CONSTANTS
# ============================================================

SAMPLE_RATE = 16000
DURATION = 4
SAMPLES = SAMPLE_RATE * DURATION

TIME_FRAMES = 126
FEATURE_DIM = 154

# CNN confidence bands (CRITICAL)
REAL_CONFIDENT_MAX = 1e-6      # extremely confident REAL
FAKE_CONFIDENT_MIN = 0.90      # confident FAKE

# ============================================================
# MODEL
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

# ============================================================
# AUDIO
# ============================================================

def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    return audio

# ============================================================
# FEATURES (UNCHANGED – TRAINING MATCH)
# ============================================================

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
        features = np.pad(
            features,
            ((0,0),(0,TIME_FRAMES - features.shape[1])),
            mode="constant"
        )

    return features.astype(np.float32)

# ============================================================
# LOAD MODEL
# ============================================================

model = build_model()
model.load_weights("model/deepfake_cnn_compat.h5")
print("✅ Model loaded")

# ============================================================
# FINAL PREDICTION (MODEL ONLY)
# ============================================================

def predict_audio(path):
    audio = load_audio(path)
    features = extract_features(audio)

    cnn_input = features[np.newaxis, ..., np.newaxis]
    prob = float(model.predict(cnn_input, verbose=0)[0][0])

    # --- Decision logic ---
    if prob >= FAKE_CONFIDENT_MIN:
        return "FAKE", prob

    if prob <= REAL_CONFIDENT_MAX:
        # Extremely confident REAL → could be high-quality clone
        return "POSSIBLE AI-GENERATED (HIGH-QUALITY)", 1 - prob

    if prob < 0.5:
        return "REAL", 1 - prob

    return "FAKE", prob

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    file_path = "test_audio/lokesh.wav"

    if not os.path.exists(file_path):
        print("File not found:", file_path)
        exit(1)

    label, confidence = predict_audio(file_path)

    print("\nAudio file :", file_path)
    print("Result     :", label)
    print("Confidence :", round(confidence * 100, 6), "%")
