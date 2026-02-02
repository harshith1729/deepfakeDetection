import os
import librosa
import numpy as np
from tqdm import tqdm

# ============================================================
# CONSTANTS
# ============================================================

SAMPLE_RATE = 16000
DURATION = 4
SAMPLES = SAMPLE_RATE * DURATION

# ============================================================
# AUDIO LOADING & STANDARDIZATION
# ============================================================

def load_audio(file_path):
    """
    Load audio file and standardize it:
    - mono
    - resample to 16kHz
    - trim/pad to 4 seconds
    """
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))

    return audio


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    return mfcc


def extract_delta_mfcc(mfcc):
    delta = librosa.feature.delta(mfcc)
    return delta


def extract_log_mel(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128)
    log_mel = librosa.power_to_db(mel)
    return log_mel


def extract_features(audio):
    """
    Combine MFCC + Delta MFCC + LogMel into one tensor
    Output: (154, time_frames)
    """
    mfcc = extract_mfcc(audio)
    delta = extract_delta_mfcc(mfcc)
    log_mel = extract_log_mel(audio)

    min_frames = min(mfcc.shape[1], delta.shape[1], log_mel.shape[1])

    mfcc = mfcc[:, :min_frames]
    delta = delta[:, :min_frames]
    log_mel = log_mel[:, :min_frames]

    features = np.vstack([mfcc, delta, log_mel])

    # Normalize
    features = (features - np.mean(features)) / (np.std(features) + 1e-6)

    return features.astype(np.float32)


# ============================================================
# PROTOCOL FILE READING (LABELS)
# ============================================================

def read_protocol(protocol_file):
    """
    Read protocol file and map filenames to labels.
    bonafide -> 1
    spoof -> 0
    """
    labels = {}

    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            file_id = parts[1]        # LA_T_1000001
            label = parts[-1]         # bonafide/spoof

            labels[file_id] = 1 if label == "bonafide" else 0

    return labels


# ============================================================
# DATASET PROCESSING
# ============================================================

def process_dataset(audio_dir, protocol_file, max_files=None):
    """
    Convert all flac files into features + labels with progress bar.
    """
    X = []
    y = []

    labels = read_protocol(protocol_file)
    files = [f for f in os.listdir(audio_dir) if f.endswith(".flac")]

    if max_files:
        files = files[:max_files]

    print(f"\n📌 Audio folder: {audio_dir}")
    print(f"📌 Total .flac files found: {len(files)}", flush=True)

    skipped = 0

    for file in tqdm(files, desc=f"Processing {os.path.basename(audio_dir)}"):
        file_id = file.replace(".flac", "")

        if file_id not in labels:
            skipped += 1
            continue

        file_path = os.path.join(audio_dir, file)

        audio = load_audio(file_path)
        features = extract_features(audio)

        X.append(features)
        y.append(labels[file_id])

    print(f"✅ Finished: {len(X)} processed | ❌ Skipped: {skipped}", flush=True)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ============================================================
# RUN PREPROCESSING (TRAIN & DEV)
# ============================================================

if __name__ == "__main__":

    # ✅ Correct root folder based on your VS Code structure
    BASE_PATH = "data/asvspoof/ASVspoof2019_LA"

    train_audio_dir = os.path.join(BASE_PATH, "ASVspoof2019_LA_train", "flac")
    train_protocol = os.path.join(BASE_PATH, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")

    dev_audio_dir = os.path.join(BASE_PATH, "ASVspoof2019_LA_dev", "flac")
    dev_protocol = os.path.join(BASE_PATH, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")

    # Print paths to verify
    print("\n================ PATH CHECK ================")
    print("TRAIN AUDIO DIR :", train_audio_dir)
    print("TRAIN PROTOCOL  :", train_protocol)
    print("DEV AUDIO DIR   :", dev_audio_dir)
    print("DEV PROTOCOL    :", dev_protocol)
    print("===========================================\n")

    # Check if files exist
    if not os.path.exists(train_audio_dir):
        raise FileNotFoundError(f"❌ Train audio folder not found: {train_audio_dir}")

    if not os.path.exists(train_protocol):
        raise FileNotFoundError(f"❌ Train protocol file not found: {train_protocol}")

    if not os.path.exists(dev_audio_dir):
        raise FileNotFoundError(f"❌ Dev audio folder not found: {dev_audio_dir}")

    if not os.path.exists(dev_protocol):
        raise FileNotFoundError(f"❌ Dev protocol file not found: {dev_protocol}")

    # Process datasets
    print("🔵 Processing TRAIN set...", flush=True)
    X_train, y_train = process_dataset(train_audio_dir, train_protocol)

    print("\n🟣 Processing DEV set...", flush=True)
    X_dev, y_dev = process_dataset(dev_audio_dir, dev_protocol)

    # ============================================================
    # SAVE OUTPUT
    # ============================================================

    os.makedirs("data", exist_ok=True)

    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)

    np.save("data/X_dev.npy", X_dev)
    np.save("data/y_dev.npy", y_dev)

    print("\n✅ Day 3 complete: Audio successfully converted to numerical features.")
    print("X_train:", X_train.shape, "| y_train:", y_train.shape)
    print("X_dev  :", X_dev.shape, "| y_dev  :", y_dev.shape)