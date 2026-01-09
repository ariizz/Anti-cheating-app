import cv2
import mediapipe as mp
import time
import json
from datetime import datetime
from typing import Optional

# --- Configuration thresholds (tune as needed) ---
LOOK_AWAY_YAW_THRESH = 20.0       # degrees
LOOK_AWAY_DURATION = 3.0          # seconds of continuous looking away
FACE_MISSING_DURATION = 5.0       # seconds of no detectable face

LOG_FILE = "incidents.log"        # local JSON Lines log file

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def log_incident(incident_type: str, details: Optional[dict] = None) -> None:
    """Append a single incident entry to the local log file."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": incident_type,
        "details": details or {},
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print("Logged incident:", entry)


def estimate_yaw_from_landmarks(landmarks, image_width: int, image_height: int) -> float:
    """
    Estimate a rough yaw angle from face landmarks.

    Uses the horizontal offset of the nose tip relative to the midpoint between the outer
    corners of the eyes. This is a simple heuristic, not a precise pose estimation.
    """
    # Indices reference MediaPipe Face Mesh canonical landmarks
    nose_tip = landmarks[1]
    left_eye_outer = landmarks[33]
    right_eye_outer = landmarks[263]

    nose_x = nose_tip.x * image_width
    left_x = left_eye_outer.x * image_width
    right_x = right_eye_outer.x * image_width

    center_eyes = (left_x + right_x) / 2.0
    offset = nose_x - center_eyes

    # Normalize by eye distance to get a pseudo-angle in degrees
    eye_distance = max(right_x - left_x, 1e-6)
    yaw_degrees = (offset / eye_distance) * 30.0
    return yaw_degrees


def run_proctoring() -> None:
    """Main loop: read webcam, detect face/head direction, log incidents."""
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    # Try to set camera properties for better compatibility
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Trying camera index 1...")
        cap.release()
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("ERROR: Could not open any webcam. Please check your camera connection.")
            return
    
    print("Camera opened successfully!")
    
    # Make preview window resizable and a bit larger
    window_name = "Proctoring Preview (Local Only)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    print(f"Preview window '{window_name}' created. Make sure it's visible!")

    # State to track durations
    look_away_start = None
    face_missing_start = None

    print("Initializing MediaPipe Face Mesh...")
    try:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("MediaPipe initialized successfully!")
    except Exception as e:
        print(f"ERROR initializing MediaPipe: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    print("Press 'q' in the camera window to quit proctoring.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera.")
            break
        
        frame_count += 1
        if frame_count == 1:
            print(f"First frame captured! Frame size: {frame.shape}")

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            results = face_mesh.process(frame_rgb)
            face_present = results.multi_face_landmarks is not None
        except Exception as e:
            print(f"ERROR processing frame: {e}")
            continue

        now = time.time()

        if face_present:
            # Reset face-missing timer
            face_missing_start = None

            landmarks = results.multi_face_landmarks[0].landmark
            yaw = estimate_yaw_from_landmarks(landmarks, w, h)

            # Simple "looking away" rule
            if abs(yaw) > LOOK_AWAY_YAW_THRESH:
                if look_away_start is None:
                    look_away_start = now
                elif now - look_away_start >= LOOK_AWAY_DURATION:
                    log_incident(
                        "LOOKING_AWAY",
                        {"yaw_degrees": yaw, "duration_sec": now - look_away_start},
                    )
                    # Reset so that a very long look generates spaced incidents
                    look_away_start = now
            else:
                # Back to normal gaze
                look_away_start = None

            # Draw landmarks and yaw for local debug preview
            mp_drawing.draw_landmarks(
                frame,
                results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,
            )
            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            # No face detected
            if face_missing_start is None:
                face_missing_start = now
            elif now - face_missing_start >= FACE_MISSING_DURATION:
                log_incident(
                    "FACE_NOT_VISIBLE",
                    {"duration_sec": now - face_missing_start},
                )
                face_missing_start = now  # reset

            # If no face, also reset head-direction timer
            look_away_start = None

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Always show the current frame
        cv2.imshow(window_name, frame)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting...")
            break

    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    run_proctoring()

