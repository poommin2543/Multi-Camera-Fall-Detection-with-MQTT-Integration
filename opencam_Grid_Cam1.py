import cv2
import numpy as np

# Open the RTSP stream
cam = cv2.VideoCapture("rtsp://CamZero:acselab123@192.168.0.11:554/stream1")

# Set the desired frame width and height
frame_width = 640
frame_height = 480

# Define source points for perspective transformation
src_pts = np.float32([
    [227, 327],  # Top-left
    [554, 331],  # Top-right
    [154, 426],  # Bottom-left
    [604, 440]   # Bottom-right
])

# Define destination points for a normalized rectangle
dst_pts = np.float32([
    [0, 0],  # Mapped to top-left corner
    [frame_width, 0],  # Mapped to top-right corner
    [0, frame_height],  # Mapped to bottom-left corner
    [frame_width, frame_height]  # Mapped to bottom-right corner
])

# Compute the forward and inverse perspective transform matrices
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)  # Inverse transformation

# Function to create grid points
def get_grid_points(rows=3, cols=6):
    points = []
    step_x = frame_width // cols
    step_y = frame_height // rows

    for i in range(1, cols):
        x = i * step_x
        points.append((x, 0))  # Top row
        points.append((x, frame_height))  # Bottom row

    for i in range(1, rows):
        y = i * step_y
        points.append((0, y))  # Left column
        points.append((frame_width, y))  # Right column

    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)

# Function to draw lines between grid points (with proper type conversion)
def draw_grid(frame, points, color=(0, 255, 0), thickness=1):
    num_points = len(points) // 2
    for i in range(num_points):
        pt1 = tuple(map(int, points[i * 2][0]))  # Convert NumPy float to int tuple
        pt2 = tuple(map(int, points[i * 2 + 1][0]))  # Convert NumPy float to int tuple
        cv2.line(frame, pt1, pt2, color, thickness)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to 640x480
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Apply the perspective transformation
    warped_frame = cv2.warpPerspective(frame, M, (frame_width, frame_height))

    # Generate grid points in the warped frame
    grid_points = get_grid_points()

    # Transform the grid points back to the original frame
    original_grid_points = cv2.perspectiveTransform(grid_points, Minv)

    # Draw grid on both frames
    draw_grid(warped_frame, grid_points)  # Grid in warped frame
    draw_grid(frame, original_grid_points)  # Grid transformed back to original frame

    # Display both frames
    cv2.imshow('Perspective View with Grid', warped_frame)
    cv2.imshow('Original Frame with Transformed Grid', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object
cam.release()
cv2.destroyAllWindows()
