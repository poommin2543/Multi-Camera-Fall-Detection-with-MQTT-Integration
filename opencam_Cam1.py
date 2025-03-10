import cv2

# Open the RTSP stream
cam = cv2.VideoCapture("rtsp://CamZero:acselab123@192.168.0.11:554/stream1")

# Set the desired frame width and height
frame_width = 640
frame_height = 480

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to 640x480
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Display the resized frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object
cam.release()
cv2.destroyAllWindows()
