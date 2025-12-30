# OpenCV OCR Firebase Digit Detection

Real-time digit detection system using OpenCV and PaddleOCR.
Detected digits are sent directly to Firebase Realtime Database.

## Features
- PaddleOCR-based digit recognition
- ROI selection
- Automatic & manual detection modes
- Firebase Realtime Database integration
- FPS monitoring

## Usage
```bash
pip install -r requirements.txt
python src/digit_detector.py

## Controls
- `s` : Manual digit detection
- `a` : Toggle automatic detection
- `r` : Select ROI
- `q` : Quit application

## Project Architecture
Camera → OpenCV + PaddleOCR → Digit Extraction → Firebase Realtime Database
