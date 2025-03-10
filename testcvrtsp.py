import cv2
import mediapipe as mp
import time

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# RTSP Camera URLs
camera_urls = [
    "rtsp://CamZero:acselab123@192.168.0.11:554/stream1",
    "rtsp://CamZero:acselab123@192.168.0.12:554/stream1",
]
camera_names = ["ห้องนั่งเล่น", "ห้องนอน"]

# Initialize VideoCapture for each camera
# caps = [cv2.VideoCapture(url) for url in camera_urls]
caps = [cv2.VideoCapture(url, cv2.CAP_FFMPEG) for url in camera_urls]


# Fall detection variables
fall_detected = [False] * len(camera_urls)
last_notification_time = [0] * len(camera_urls)
cooldown_time = 10  # Cooldown time in seconds

while True:
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Camera {camera_names[i]} is offline.")
            cap.open(camera_urls[i])
            continue

        ret, frame = cap.read()
        if not ret:
            print(f"Camera {camera_names[i]}: No frame received.")
            continue

        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y

            # Fall detection logic
            if abs(nose_y - left_foot_y) < 0.1:  # Adjust threshold as needed
                current_time = time.time()
                if not fall_detected[i] or (current_time - last_notification_time[i] > cooldown_time):
                    print(f"Fall detected in {camera_names[i]}!")
                    fall_detected[i] = True
                    last_notification_time[i] = current_time
            else:
                fall_detected[i] = False

        cv2.imshow(f'Camera {camera_names[i]}', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
