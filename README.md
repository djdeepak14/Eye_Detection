# 👁️ Eye Detection using OpenCV

A simple real-time Eye Detection project built using Python and OpenCV.  
This project detects human eyes from a webcam feed and highlights them using bounding boxes.

---

## 📌 Project Overview

This project demonstrates how traditional computer vision techniques can be used to detect eyes in real time.

It uses:

- Python
- OpenCV
- Haar Cascade Classifier

The system captures frames from a webcam, processes them in grayscale, and detects eyes using a pre-trained Haar cascade model.

---

## 🧠 How It Works

1. The webcam captures live video.
2. Each frame is converted to grayscale.
3. The Haar Cascade eye classifier scans the image.
4. When eyes are detected, rectangles are drawn around them.
5. The processed video is displayed in real time.

Press **Q** to exit the application.

---

## 🖥️ Requirements

Make sure you have:

- Python 3.x
- OpenCV

Install OpenCV using:

```bash
pip install opencv-python
```

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/djdeepak14/Eye_Detection.git
```

2. Navigate to the project folder:

```bash
cd Eye_Detection
```

3. Run the script:

```bash
python Eye_detection.py
```

Your webcam will open and start detecting eyes.

---

## 📂 Project Structure

```
Eye_Detection/
│
├── Eye_detection.py
├── haarcascade_eye.xml (if included)
└── README.md
```

---

## 💡 Features

- Real-time eye detection
- Lightweight and fast
- Easy to understand code
- Beginner-friendly computer vision project

---

## 🔍 Possible Improvements

- Add face detection before eye detection for better accuracy
- Implement blink detection
- Add eye tracking
- Save detected eye coordinates
- Convert into a GUI application

---

## 👨‍💻 Author

Deepak Khanal  

---

## ⭐ Support

If you found this project helpful, consider giving it a star on GitHub!
