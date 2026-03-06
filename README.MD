# DeepSense: Advanced Deepfake Detection

DeepSense is a powerful, AI-driven tool designed to detect deepfake images and videos. Utilizing state-of-the-art deep learning architectures, it provides high-accuracy analysis to help distinguish between real and manipulated media.

## 🚀 Features

- **Multi-Media Support**: Analyze both images (JPG, PNG, WebP, etc.) and videos (MP4, AVI, MOV).
- **Advanced Model**: Built on **EfficientNet-B0**, fine-tuned for high-precision forgery detection.
- **TTA (Test-Time Augmentation)**: Enhances accuracy by aggregating multiple views of the same image.
- **Video Sampling**: Intelligently samples frames and aggregates predictions for reliable video analysis.
- **Web Interface**: User-friendly Flask-based dashboard for easy file uploads and real-time results.
- **Visual Previews**: Live previews of analyzed images and sampled video frames.

## 🛠️ Tech Stack

- **Core**: Python 3.x
- **Deep Learning**: PyTorch, Torchvision
- **Computer Vision**: OpenCV, PIL (Pillow)
- **Web Framework**: Flask
- **Data Science**: NumPy, Scikit-learn (for training)

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/simpharsh/deepsense.git
   cd deepsense
   ```

2. **Provide the model**:
   Make sure the `best_model.pth` file is located in the root directory.

---
*Disclaimer: This tool is for educational and research purposes. Always verify sensitive content manually.*