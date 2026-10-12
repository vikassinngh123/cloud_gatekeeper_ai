# GateKeeper AI 

GateKeeper AI is a computer vision security prototype that combines virtual fencing, object detection, and facial recognition to automate access control and intrusion alerts. 

Instead of relying on physical barriers, this system allows operators to draw digital boundaries over video feeds. It uses AI to monitor these zones, cross-reference intruders against a database of authorized personnel, and instantly trigger email alerts if a breach occurs.

## Key Features
* **Interactive Virtual Fencing:** Draw custom, polygonal restricted zones directly over video frames to define monitored perimeters.
* **Intrusion Detection:** Utilizes Ultralytics YOLOv8 for high-accuracy, real-time person detection and foot-point tracking.
* **Biometric Access Control:** Integrates OpenCV YuNet and SFace for facial detection and 128D feature extraction, ensuring registered employees do not trigger false alarms.
* **Automated SMTP Alerts:** Background daemon that instantly dispatches email notifications to security personnel the moment an unauthorized individual breaches the zone.
* **Video Enhancement:** Includes optional CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing for low-light or low-contrast footage.

## Tech Stack
* **Core Logic & UI:** Python, Streamlit
* **Computer Vision:** OpenCV (cv2)
* **Object Detection:** Ultralytics (YOLOv8n)
* **Face Detection/Recognition:** OpenCV YuNet & SFace (ONNX models)
* **Alerts:** Python `smtplib`

## Installation and Setup
To run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/vikassinngh123/cloud_gatekeeper_ai.git](https://github.com/vikassinngh123/cloud_gatekeeper_ai.git)
   cd cloud_gatekeeper_ai
   ```

2. **Install dependencies:**
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run demo_app.py
   ```

## Usage Workflow
1. **Configure Alerts:** Enter your SMTP credentials (e.g., Gmail and Google App Password) in the sidebar to enable automated email notifications.
2. **Set the Perimeter:** Upload a security video, scrub to a clear frame, and draw a restricted zone using the canvas tool. 
3. **Enroll Personnel:** Register authorized individuals by uploading their photos so the system recognizes them as safe.
4. **Monitor:** Start the live analysis. The system will track movement, ignore authorized faces, and flag unknown intruders entering the zone.
