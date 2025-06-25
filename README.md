# ASL Real-Time Recognition

![Demonstration](assets/asl_demonstration.gif)

A real-time American Sign Language (ASL) alphabet recognition system using [PyTorch, MediaPipe, OpenCV, etc.]. 

## 👀 Features
- 🪄 **Real-time detection** (30+ FPS on a standard webcam)
- 👋 **Robust to skin tones** (trained on multiple-sourced data)
- 🏙️ **Background-invariant** (works in cluttered environments)
- 🔠 **Supports 26 ASL letters** (A-Z)

## 🔧 Technologies used
- **Hand Tracking**: [MediaPipe](https://mediapipe.dev/) (via `cvzone.HandTrackingModule`)
- **Model Inference**: PyTorch
- **Camera Processing**: OpenCV

## ⚙️  Installation
### Prerequisites
- Python 3.11+
- Webcam

### 🚀 Steps
```bash
# Clone the repo
git clone https://github.com/mrariv/asl_recognition.git
cd asl_recognition

# Install dependencies
pip install -r requirements.txt```

## 🍵 Usage
```bash
python run.py # Default webcam (camera ID 0)

## 🙏 Contributors
Thanks to all the amazing people who contributed! See [CONTRIBUTORS.md](CONTRIBUTORS.md) for details.

## 📜 License
This project is licensed under the [MIT License](LICENSE).
