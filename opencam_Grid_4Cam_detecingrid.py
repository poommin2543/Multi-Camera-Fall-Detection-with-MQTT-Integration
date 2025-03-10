import cv2
import mediapipe as mp
import threading
import numpy as np
import time

# ตั้งค่ากล้อง RTSP
camera_ips = ["192.168.0.11", "192.168.0.12", "192.168.0.13", "192.168.0.14"]
rtsp_urls = [f"rtsp://CamZero:acselab123@{ip}:554/stream1" for ip in camera_ips]
camera_names = ["Camera 1", "Camera 2", "Camera 3", "Camera 4"]

# ขนาดเฟรม
frame_width, frame_height = 640, 480
num_cameras = len(rtsp_urls)

# Shared variable for frames
frames = [np.zeros((frame_height, frame_width, 3), dtype=np.uint8)] * num_cameras
lock = threading.Lock()

# Fall Detection Tracking
fall_timers = [0] * num_cameras
fall_threshold = 1.5  # วินาทีในการยืนยันว่าล้ม

# กำหนด Perspective Transform สำหรับแต่ละกล้อง
src_points = [
    np.float32([[227, 327], [554, 331], [154, 426], [604, 440]]),  # Camera 1
    np.float32([[586, 328], [71, 324], [511, 187], [141, 179]]),  # Camera 2
    np.float32([[585, 379], [48, 372], [501, 240], [122, 227]]),  # Camera 3
    np.float32([[152, 272], [481, 266], [88, 379], [539, 376]])   # Camera 4
]
dst_pts = np.float32([
    [0, 0], [frame_width, 0], [0, frame_height], [frame_width, frame_height]
])

# คำนวณ Transformation Matrix
perspective_matrices = [cv2.getPerspectiveTransform(src, dst_pts) for src in src_points]
inverse_matrices = [cv2.getPerspectiveTransform(dst_pts, src) for src in src_points]

# สร้าง Grid Points
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

# วาดเส้นกริด
def draw_grid(frame, points, color=(0, 255, 0), thickness=1):
    num_points = len(points) // 2
    for i in range(num_points):
        pt1 = tuple(map(int, points[i * 2][0]))
        pt2 = tuple(map(int, points[i * 2 + 1][0]))
        cv2.line(frame, pt1, pt2, color, thickness)
def get_grid_cell(px, py, rows=3, cols=6):
    """
    คำนวณว่าจุด (px, py) อยู่ในแถวและคอลัมน์ที่เท่าไหร่ของกริด
    """
    cell_width = frame_width // cols
    cell_height = frame_height // rows

    col = int(px // cell_width) + 1  # คอลัมน์เริ่มจาก 1
    row = int(py // cell_height) + 1  # แถวเริ่มจาก 1

    return row, col

# เพิ่มตัวแปรเก็บสถานะการแจ้งเตือนของแต่ละกล้อง
fall_detected = [False] * num_cameras  

# ฟังก์ชันประมวลผลกล้องแต่ละตัว
def process_camera(index, rtsp_url):
    global frames, fall_timers
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # คำนวณค่าของกล้องแต่ละตัว
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
            original_grid_points = cv2.perspectiveTransform(grid_points, Minv)

            # วาด Grid บนเฟรม
            draw_grid(warped_frame, grid_points)
            draw_grid(frame, original_grid_points)

            # ตรวจจับบุคคลด้วย MediaPipe Pose
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            processed_frame = frame.copy()
            bbox_color = (0, 255, 0)  # สีเริ่มต้นของ Bounding Box (เขียว)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # คำนวณ Bounding Box รอบตัวบุคคล
                x_min = int(min(l.x for l in landmarks) * frame_width)
                y_min = int(min(l.y for l in landmarks) * frame_height)
                x_max = int(max(l.x for l in landmarks) * frame_width)
                y_max = int(max(l.y for l in landmarks) * frame_height)

                # ป้องกันค่าที่เกินขนาดภาพ
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(frame_width, x_max)
                y_max = min(frame_height, y_max)

                # วาดกรอบรอบตัวบุคคล
                cv2.rectangle(processed_frame, (x_min, y_min), (x_max, y_max), bbox_color, 3)

                # คำนวณจุดศูนย์กลางของบุคคล
                # person_center = np.array([[[(x_min + x_max) / 2, (y_min + y_max) / 2]]], dtype=np.float32)
                nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x * frame_width
                nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * frame_height
                hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2 * frame_width
                hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2 * frame_height

                person_center = np.array([[[ (nose_x + hip_x) / 2, (nose_y + hip_y) / 2 ]]], dtype=np.float32)
                
                # transformed_center = cv2.perspectiveTransform(person_center, Minv)

                # Debug: แสดงจุดศูนย์กลางของบุคคลและจุดกริด
                cv2.circle(processed_frame, tuple(map(int, person_center[0][0])), 5, (255, 0, 0), -1)  # จุดศูนย์กลางบุคคล
                for pt in original_grid_points:
                    cv2.circle(processed_frame, tuple(map(int, pt[0])), 3, (0, 255, 255), -1)  # จุดกริด

                # หาขอบเขตของกริดจาก original_grid_points
                grid_x_min, grid_y_min = np.min(original_grid_points, axis=0)[0]
                grid_x_max, grid_y_max = np.max(original_grid_points, axis=0)[0]

                # พิกัดศูนย์กลางของบุคคล
                px, py = person_center[0][0]

                # ตรวจสอบว่าศูนย์กลางของบุคคลอยู่ภายใน Grid หรือไม่
                is_inside_grid = (grid_x_min <= px <= grid_x_max) and (grid_y_min <= py <= grid_y_max)
                # print(is_inside_grid)
                # # ถ้าอยู่นอกกริด ให้รีเซ็ต fall_timers และข้ามการตรวจจับการล้ม
                if not is_inside_grid:
                    fall_timers[index] = 0  # รีเซ็ต Timer
                    with lock:
                        frames[index] = processed_frame  # อัปเดตเฟรมแม้ไม่ตรวจจับการล้ม
                    continue


                # ตรวจจับการล้ม
                nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * frame_height
                hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + 
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2 * frame_height

                # ตรวจจับการล้ม
                if abs(nose_y - hip_y) < 80:  # ถ้าระยะห่างระหว่างจมูกและสะโพกสั้น แสดงว่ากำลังล้ม
                    if fall_timers[index] == 0:
                        fall_timers[index] = time.time()
                    elif time.time() - fall_timers[index] > fall_threshold:
                        bbox_color = (0, 0, 255)  # เปลี่ยนสีเป็นแดงเมื่อเกิดการล้ม
                        cv2.rectangle(processed_frame, (x_min, y_min), (x_max, y_max), bbox_color, 3)
                        cv2.putText(processed_frame, "LYING DETECTED!", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                        # แจ้งเตือนเฉพาะครั้งแรกของการล้ม
                        if not fall_detected[index]:
                            print(f"🚨 แจ้งเตือน: พบการล้มจาก {camera_names[index]}")
                            fall_detected[index] = True  # ตั้งค่าเป็นเคยแจ้งเตือนแล้ว
                else:
                    fall_timers[index] = 0  # รีเซ็ต Timer ถ้าบุคคลกลับมายืน
                    fall_detected[index] = False  # รีเซ็ตสถานะแจ้งเตือน

            with lock:
                frames[index] = processed_frame

        cam.release()
        time.sleep(2)


threads = [threading.Thread(target=process_camera, args=(i, rtsp_urls[i]), daemon=True) for i in range(num_cameras)]
for t in threads: t.start()

while True:
    with lock:
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        combined_frame = np.vstack((top_row, bottom_row))

    cv2.imshow("Multi-Camera View with Grid & Fall Detection", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
