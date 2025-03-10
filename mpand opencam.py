import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open RTSP stream with FFMPEG for better stability
rtsp_url = "rtsp://CamZero:acselab123@192.168.0.14:554/stream1"
cam = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# Set fixed frame size (640x480)
frame_width, frame_height = 640, 480

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to 640x480
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Convert frame to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Draw landmarks on frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
