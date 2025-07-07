# Human Activity Recognition using OpenCV & ResNet

This project performs real-time **Human Activity Recognition (HAR)** using a pre-trained ResNet-34 model from the Kinetics dataset. It uses OpenCV for video capture and processing, and ONNX for loading the deep learning model.

## 📽️ Demo
The model can predict actions such as walking, clapping, jumping, etc., from a video or webcam feed in real-time.

## 🔧 Features
- Uses `resnet-34_kinetics.onnx` model trained on the [Kinetics dataset].
- Accepts both **video file input** and **live webcam** input.
- Smooth prediction by sampling frames using a `deque` buffer.
- Displays predicted activity label in real-time.

## 📂 Project Structure

project/
├── model/
│ ├── resnet-34_kinetics.onnx
│ └── action_recognition_kinetics.txt
├── recognise_human_activity.py
└── README.md

bash
Copy
Edit

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
 2. Install Dependencies
Make sure Python and OpenCV are installed:

bash
Copy
Edit
pip install opencv-python numpy
### 3. Run the Code
bash
Copy
Edit
python har_recognition.py
Press q to exit the video window.

📄 Files Explained
recognise_human_activity.py – Main Python script for inference.

resnet-34_kinetics.onnx – Pre-trained action recognition model.

action_recognition_kinetics.txt – List of 400 activity labels (one per line).
