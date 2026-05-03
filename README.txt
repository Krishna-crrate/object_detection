🧠 Object Detection System (YOLOv3)

Python • OpenCV • YOLOv3 • Tkinter GUI

A real-time object detection system built using YOLOv3 and OpenCV with
an interactive Tkinter-based GUI. The application supports both live
webcam detection and video file detection with bounding boxes and confidence scores.

🎯 Key Features:
  🎥 Real-Time Detection – Detect objects using live webcam feed
  📂 Video File Detection – Upload and analyze recorded videos
  🧠 YOLOv3 Model – Pre-trained deep learning model for accurate detection
  🖥️ GUI Interface – Built with Tkinter for easy interaction
  🏷️ Object Labeling – Displays object name with confidence score
  ⚡ Multithreading Support – Smooth performance without UI freezing
  🎞️ Frame Processing – Optimized resizing and detection pipeline
  
🏗️ How It Works
  Input (Webcam / Video)
        ↓
  Frame Capture (OpenCV)
        ↓
  Preprocessing (Resize + Blob Conversion)
        ↓
  YOLOv3 Detection (cv2.dnn)
        ↓
  Non-Max Suppression (NMS)
        ↓
  Output (Bounding Boxes + Labels)

🚀 Quick Start

🔧 Prerequisites
    Python 3.7+
    OpenCV
    NumPy
    Pillow

⚙️ Installation
# Clone repository
git clone https://github.com/your-username/object-detection.git
cd object-detection

# Install dependencies
pip install opencv-python numpy pillow

▶️ Run the Project
python mango.py

🖥️ Application Features (GUI)
    📌 VIDEO DETECTION → Detect objects from uploaded video
    📌 Live Cam Detection → Real-time webcam detection
    📌 Upload Video → Select video file
    📌 Quit → Close application

📁 Project Structure
object-detection/
│
├── mango.py              # Main application file
├── yolov3.cfg            # YOLO configuration
├── yolov3.weights        # Pre-trained weights
├── coco.names            # Class labels
├── logo.png              # GUI background image
├── README.md

🧪 Detection Details
     Confidence Threshold: 0.3
     NMS Threshold: 0.4
     Input Size: 416x416
     Frame Resize: 640x480

🛠️ Tech Stack
Category	        Technology
Language	        Python
Computer Vision	    OpenCV (cv2.dnn)
Deep Learning	    YOLOv3
GUI	                Tkinter
Image Handling	    Pillow

📌 Future Improvements
    🚀 Upgrade to YOLOv8 for higher accuracy
    🎯 Add object tracking (SORT / DeepSORT)
    🌐 Web-based interface (Flask/React)
    📊 Detection analytics dashboard
    📱 Mobile deployment

IMPORTANT:
📥 Download YOLO Weights
Download from:
👉 https://pjreddie.com/media/files/yolov3.weights
Place it inside your project folder.


👨‍💻 Author
Venkat Krishna

GitHub: https://github.com/Krishna-crrate
LinkedIn: www.linkedin.com/in/venkatakrishna2004

⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
