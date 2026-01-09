import opencv as oc
import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam (0 is usually the default camera)
video_capture = cv2.VideoCapture(0)

# cv2.namedWindow('Face Detection Project', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Face Detection Project', 1280, 720)

print("Press 'q' to quit the application.")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert to grayscale (OpenCV works better with gray images for detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Face Detection Project', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
import time

LOOK_AWAY_YAW_THRESH = 2.0  # degrees (change as necessary)
LOOK_AWAY_DURATION = 2.0     # seconds

# Additional state variables
look_away_start = None

def estimate_yaw_from_face(x, w, frame_width):
    """
    Very rough estimation: if the face center is far away from the image center,
    assume turned head.
    """
    face_center_x = x + w / 2
    image_center_x = frame_width / 2
    offset = face_center_x - image_center_x
    # Normalize: relative to face width, gives a rough "angle" in degrees
    # Tweak multiplier as necessary for sensitivity
    eye_distance = w
    pseudo_yaw = (offset / max(eye_distance, 1e-3)) * 30
    return pseudo_yaw

alert_message_on = False
alert_start_time = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    frame_h, frame_w = frame.shape[:2]
    face_detected = False
    yaw_value = 0
    alert_this_frame = False

    for (x, y, w, h) in faces:
        face_detected = True
        yaw_value = estimate_yaw_from_face(x, w, frame_w)

        # Detect looking away
        if abs(yaw_value) > LOOK_AWAY_YAW_THRESH:
            if look_away_start is None:
                look_away_start = time.time()
            elif time.time() - look_away_start > LOOK_AWAY_DURATION:
                # Alert: draw the box in red and set alert
                box_color = (0, 0, 255)
                alert_message_on = True
                alert_this_frame = True
                alert_start_time = time.time()
            else:
                box_color = (0, 255, 0)
        else:
            look_away_start = None
            box_color = (0, 255, 0)
            alert_message_on = False
            alert_this_frame = False

        # If user just returned to normal, turn off alert after 2s
        if alert_message_on and not alert_this_frame:
            if alert_start_time and (time.time() - alert_start_time > 2):
                alert_message_on = False

        # Draw rectangle (red if alert, green otherwise)
        cv2.rectangle(frame, (int(x*1.5), int(y*1.5)), (int((x+w)*1.5), int((y+h)*1.5)), 
                      (0, 0, 255) if alert_message_on else box_color, 2)

        # Display yaw value for debug
        cv2.putText(
            frame,
            f"Yaw: {yaw_value:.1f}",
            (int(x*1.5), int((y-10)*1.5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        break  # Only monitor the first (largest) detected face

    if not face_detected:
        look_away_start = None
        alert_message_on = False

    # Draw alert message if triggered
    if alert_message_on:
        cv2.putText(
            frame,
            "ALERT: LOOKING AWAY !",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3,
        )

    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Face Detection Project', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()