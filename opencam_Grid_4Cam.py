import cv2
import mediapipe as mp
import threading
import numpy as np
import time

# RTSP Camera Information
camera_ips = ["192.168.0.11", "192.168.0.12", "192.168.0.13", "192.168.0.14"]
rtsp_urls = [f"rtsp://CamZero:acselab123@{ip}:554/stream1" for ip in camera_ips]
camera_names = ["Camera 1", "Camera 2", "Camera 3", "Camera 4"]

# Frame size
frame_width, frame_height = 640, 480
num_cameras = len(rtsp_urls)

# Shared variable for frames
frames = [np.zeros((frame_height, frame_width, 3), dtype=np.uint8)] * num_cameras
lock = threading.Lock()

# Fall detection tracking
fall_timers = [0] * num_cameras
fall_threshold = 1.5  # Seconds to confirm lying down

# Perspective transformation points per camera
src_points = [
    np.float32([[227, 327], [554, 331], [154, 426], [604, 440]]),  # Camera 1
    np.float32([[586, 328], [71, 324], [511, 187], [141, 179]]),  # Camera 2
    np.float32([[585, 379], [48, 372], [501, 240], [122, 227]]),  # Camera 3
    np.float32([[152, 272], [481, 266], [88, 379], [539, 376]])   # Camera 4
]
dst_pts = np.float32([
    [0, 0], [frame_width, 0], [0, frame_height], [frame_width, frame_height]
])

# Precompute transformation matrices
perspective_matrices = [cv2.getPerspectiveTransform(src, dst_pts) for src in src_points]
inverse_matrices = [cv2.getPerspectiveTransform(dst_pts, src) for src in src_points]

# Function to generate grid points
def get_grid_points(rows=3, cols=6):
    points = []
    step_x = frame_width // cols
    step_y = frame_height // rows
    for i in range(1, cols):
        x = i * step_x
        points.append((x, 0))
        points.append((x, frame_height))
    for i in range(1, rows):
        y = i * step_y
        points.append((0, y))
        points.append((frame_width, y))
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)

# Function to draw grid
def draw_grid(frame, points, color=(0, 255, 0), thickness=1):
    num_points = len(points) // 2
    for i in range(num_points):
        pt1 = tuple(map(int, points[i * 2][0]))
        pt2 = tuple(map(int, points[i * 2 + 1][0]))
        cv2.line(frame, pt1, pt2, color, thickness)

# Process camera feeds with fall detection & grid overlay
def process_camera(index, rtsp_url):
    global frames, fall_timers
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Camera transformation matrices
    M = perspective_matrices[index]
    Minv = inverse_matrices[index]
    grid_points = get_grid_points()

    while True:
        cam = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not cam.isOpened():
            print(f"⚠️ {camera_names[index]}: Failed to open stream, retrying in 5s...")
            time.sleep(5)
            continue

        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                print(f"⚠️ {camera_names[index]}: Connection lost, reconnecting...")
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            warped_frame = cv2.warpPerspective(frame, M, (frame_width, frame_height))

            # Transform grid points to original perspective
            original_grid_points = cv2.perspectiveTransform(grid_points, Minv)

            # Draw grid
            draw_grid(warped_frame, grid_points)
            draw_grid(frame, original_grid_points)

            # Convert to RGB for pose processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            processed_frame = frame.copy()
            bbox_color = (0, 255, 0)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                x_min = min(l.x for l in landmarks) * frame_width
                y_min = min(l.y for l in landmarks) * frame_height
                x_max = max(l.x for l in landmarks) * frame_width
                y_max = max(l.y for l in landmarks) * frame_height
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

                mp_drawing.draw_landmarks(processed_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * frame_height
                left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame_height
                right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame_height
                left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height
                right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height

                hip_y = (left_hip_y + right_hip_y) / 2
                shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
                shoulder_diff = abs(left_shoulder_y - right_shoulder_y)

                if abs(nose_y - hip_y) < 40 and shoulder_diff < 20:
                    if fall_timers[index] == 0:
                        fall_timers[index] = time.time()
                    elif time.time() - fall_timers[index] > fall_threshold:
                        bbox_color = (0, 0, 255)
                        cv2.putText(processed_frame, "🚨 LYING DETECTED!", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        print(f"⚠️ {camera_names[index]}: LYING DETECTED!")
                else:
                    fall_timers[index] = 0

                cv2.rectangle(processed_frame, (x_min, y_min), (x_max, y_max), bbox_color, 3)

            with lock:
                frames[index] = processed_frame

        cam.release()
        time.sleep(2)

# Start threads
threads = []
for i in range(num_cameras):
    thread = threading.Thread(target=process_camera, args=(i, rtsp_urls[i]), daemon=True)
    thread.start()
    threads.append(thread)

# Display all camera feeds
while True:
    with lock:
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        combined_frame = np.vstack((top_row, bottom_row))

    cv2.imshow("Multi-Camera View with Grid & Fall Detection", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
