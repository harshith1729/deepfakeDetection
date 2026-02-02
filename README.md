# 🔐 Deepfake Voice Detection IoT System

A Deep Learning–based **Deepfake Voice Detection System** integrated with an IoT-style decision logic and a Streamlit web interface.  
The system detects whether a given voice sample is **Genuine (Human)** or **Deepfake (Spoofed)** and activates a **fallback mechanism** when prediction confidence is low.

---

## 📌 Problem Statement

With the rapid rise of AI-generated voices, **voice spoofing and deepfake attacks** pose serious threats to authentication systems, smart locks, and voice-controlled IoT devices.

This project aims to:
- Detect deepfake/spoofed voice samples using Deep Learning
- Measure prediction confidence
- Trigger a **fallback system** if confidence drops below a defined threshold
- Provide a clean, interactive interface for real-time testing

---

## 🎯 Key Features

- 🎙️ Audio recording directly from browser (Streamlit)
- 🧠 Deep Learning–based voice classification
- 📊 Confidence score–based decision making
- ⚠️ Fallback system if confidence < 60%
- 🚫 No dataset or large files pushed to GitHub (industry best practice)
- 🧪 Tested on ASVspoof dataset
- 🖥️ Simple, clean Streamlit UI

---

## 🏗️ System Architecture

<img width="1536" height="1024" alt="deepfakeArchitecture" src="https://github.com/user-attachments/assets/bcb716df-a7b9-41a8-ae5b-804d38bc0ad6" />

---

## 🧠 Model Overview

- Framework: **TensorFlow / Keras**
- Input: Processed audio features (MFCC / Mel Spectrogram)
- Output:
  - `Genuine` (Real Human Voice)
  - `Deepfake` (Spoofed / AI-generated Voice)
- Confidence-based decision logic applied post prediction

> ⚠️ Model weights are intentionally **not uploaded** due to size and best practices.

---

## 📁 Project Structure

```
DEEPFAKE_VOICE_IOT/
│
├── api/
├── audio_tools/
│
├── data/
│   ├── asvspoof/          (ignored)
│   ├── X_dev.npy          (ignored)
│   ├── X_train.npy        (ignored)
│   ├── y_dev.npy          (ignored)
│   ├── y_train.npy        (ignored)
│   └── README.md          (kept)
│
├── iot/
├── model/
│
├── preprocessing/
│   └── preprocess_audio.py
│
├── test_audio/            (ignored)
│   ├── deepfake.wav
│   ├── goutham.wav
│   └── ...
│
├── venv/                  (ignored)
│
├── .gitignore
├── requirements.txt
├── streamlit_app.py
├── test.ipynb             (optional)
└── test.py                (optional)
```

> Files and folders marked as **(ignored)** are excluded using `.gitignore`  
> to keep the repository clean and within GitHub size limits.



> These `.npy` files are generated during preprocessing and **must not be committed to GitHub**.

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd deepfake_voice_iot
## 🛠️ Environment Setup

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
```

For Windows:

```bash
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

## ⚙️ Fallback System Logic

- Model outputs prediction + confidence score  
- Threshold set at **60%**  

If confidence < 60%:
- Result marked as **Fallback Triggered**
- Prevents unsafe decision making  

This simulates real-world IoT security systems.

---

## 📊 Output Examples

| Prediction | Confidence | System Action |
|-----------|------------|---------------|
| Genuine   | 92%        | Accept        |
| Deepfake  | 88%        | Reject        |
| Genuine   | 45%        | ⚠️ Fallback Triggered |

---

## 🚫 Why Dataset & Models Are Not Uploaded

- GitHub file size limits  
- Industry best practices  
- Encourages reproducibility  
- Prevents repository bloat  

Instead:
- Dataset instructions provided  
- Model can be retrained or loaded externally  

---

## 🧪 Technologies Used

- Python  
- TensorFlow / Keras  
- Librosa  
- NumPy  
- Streamlit  
- ASVspoof Dataset  

---

## 🎓 Academic Relevance

Suitable for:
- Minor Project  
- Final Year Project (with extension)  
- AI + IoT demonstrations  

Covers:
- Deep Learning  
- Audio Signal Processing  
- Security Systems  
- Confidence-based decision making  

---

## 🚀 Future Enhancements

- ESP32 hardware integration  
- Real-time microphone streaming  
- Multi-language voice detection  
- Edge-device inference  
- Cloud-based model updates  

---

## 👤 Author

Harshith  
B.Tech Student  
AI / Machine Learning Enthusiast


