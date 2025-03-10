# 📌 Multi-Camera Fall Detection with MQTT Integration

![Multi-Camera](https://via.placeholder.com/800x400.png?text=Multi-Camera+Fall+Detection)

🚀 **Real-time multi-camera fall detection system** using **OpenCV**, **MediaPipe**, and **MQTT** for IoT-based alerts.

---

## **📖 Overview**
This project processes **RTSP video streams** from multiple cameras, applies **perspective transformation**, overlays a **grid-based mapping**, and detects **falls** using **pose estimation**.

When a fall is detected, an **MQTT notification** is published to a specified topic, allowing **real-time alerts** for monitoring systems.

## **✨ Features**
✅ Multi-camera **RTSP stream processing**  
✅ **Perspective transformation** for accurate localization  
✅ **MediaPipe Pose** for human pose tracking  
✅ **Fall detection** with adjustable sensitivity  
✅ **Grid-based localization** to determine position  
✅ **MQTT integration** for real-time notifications  
✅ **Automatic reconnection** for unstable RTSP streams  

---

## **🛠️ Installation & Setup**
### **1️⃣ Clone this repository**
```bash
git clone https://github.com/poommin2543/Multi-Camera-Fall-Detection-with-MQTT-Integration.git
cd Multi-Camera-Fall-Detection-with-MQTT-Integration
```

### **2️⃣ Install dependencies**
Make sure you have **Python 3.8+** installed, then run:
```bash
pip install opencv-python mediapipe numpy paho-mqtt
```

### **3️⃣ Configure MQTT & RTSP Cameras**
Update the `broker_address`, `username`, `password`, and `camera_ips` in the script:
```python
broker_address = "mqttlocal.roverautonomous.com"
username = "rover"
password = "rover"
camera_ips = ["192.168.0.11", "192.168.0.12", "192.168.0.13", "192.168.0.14"]
```

### **4️⃣ Run the Program**
```bash
python fall_detection.py
```
The system will start **streaming**, processing, and sending **MQTT alerts** if a fall is detected.

---

## **🔧 Configuration**
### **📡 MQTT Settings**
Modify the MQTT topic for fall detection alerts:
```python
topic_name = "contro/status"
```

### **📷 Camera Settings**
Define RTSP camera streams:
```python
rtsp_urls = [f"rtsp://CamZero:acselab123@{ip}:554/stream1" for ip in camera_ips]
```

### **📏 Fall Detection Sensitivity**
Adjust the **fall detection threshold**:
```python
fall_threshold = 1.5  # Time in seconds to confirm a fall
```

---

## **🖥️ UI Preview**
The system will display a **4-camera grid** with fall detection highlights:
- 🟩 **Green** bounding box = Standing  
- 🟥 **Red** bounding box = Fall detected  
- 📢 **MQTT Alert** sent on fall detection  

---

## **📌 Future Enhancements**
🔹 **Support for more cameras**  
🔹 **Web-based dashboard** for remote monitoring  
🔹 **Custom alert configurations** (SMS, Email, etc.)  

---

## **🤝 Contributing**
1. Fork the repository  
2. Create a new branch (`git checkout -b feature-branch`)  
3. Commit your changes (`git commit -m "Add new feature"`)  
4. Push to the branch (`git push origin feature-branch`)  
5. Open a Pull Request  

---

## **📜 License**
MIT License. Feel free to use and modify.  

---

## **📧 Contact**
For any questions or collaborations, feel free to reach out:  
📩 **Email:** poommin2543@gmail.com

