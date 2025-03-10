import cv2
import mediapipe as mp
import threading
import numpy as np
import time

# Define RTSP camera URLs
camera_ips = ["192.168.0.11", "192.168.0.12", "192.168.0.13", "192.168.0.14"]
rtsp_urls = [f"rtsp://CamZero:acselab123@{ip}:554/stream1" for ip in camera_ips]
camera_names = ["Camera 1", "Camera 2", "Camera 3", "Camera 4"]

# Set fixed frame size (640x480)
frame_width, frame_height = 640, 480
num_cameras = len(rtsp_urls)

# Shared variable for frames
frames = [np.zeros((frame_height, frame_width, 3), dtype=np.uint8)] * num_cameras
lock = threading.Lock()

def process_camera(index, rtsp_url):
    """ Function to process each RTSP camera in a separate thread with individual Pose instances """
    global frames
    # Create separate Pose instance for each thread
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    while True:
        cam = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce latency

        if not cam.isOpened():
            print(f"⚠️ {camera_names[index]}: Failed to open stream, retrying in 5s...")
            time.sleep(5)
            continue

        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                print(f"⚠️ {camera_names[index]}: Connection lost, reconnecting...")
                break  # Exit loop to reconnect

            # Resize and process frame
            frame = cv2.resize(frame, (frame_width, frame_height))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Copy the frame to avoid modifying shared memory
            processed_frame = frame.copy()

            # Draw landmarks only if detected
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(processed_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Update shared frame buffer safely
            with lock:
                frames[index] = processed_frame

        cam.release()  # Ensure resources are freed before reconnecting
        time.sleep(2)  # Wait before reconnecting

# Start threads
threads = []
for i in range(num_cameras):
    thread = threading.Thread(target=process_camera, args=(i, rtsp_urls[i]), daemon=True)
    thread.start()
    threads.append(thread)

# Display all camera feeds in a grid
while True:
    with lock:
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        combined_frame = np.vstack((top_row, bottom_row))

    cv2.imshow("Multi-Camera View", combined_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
