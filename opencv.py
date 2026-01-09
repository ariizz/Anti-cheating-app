import cv2
import time
import numpy as np
from collections import deque

# Load the pre-trained face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Initialize the webcam (0 is usually the default camera)
video_capture = cv2.VideoCapture(0)

# Set camera resolution for better detection
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Detection thresholds - more precise values
FACE_CENTER_THRESHOLD = 0.18  # 18% deviation from center is acceptable
LOOK_AWAY_DURATION = 2.0       # seconds before alert triggers
MIN_FACE_SIZE = 80            # minimum face size for reliable detection
EYE_DETECTION_RATIO = 0.45    # eyes should be in upper 45% of face ROI
FACE_ANGLE_THRESHOLD = 12     # degrees - more precise threshold
EYE_DETECTION_CONFIDENCE = 2  # minimum number of eyes to detect for normal state

# Temporal smoothing parameters
SMOOTHING_WINDOW = 5          # number of frames for smoothing
CONFIDENCE_THRESHOLD = 0.7    # confidence level for distraction detection

import urllib.request
import json
from datetime import datetime
import threading

# Configuration for Dashboard Connection
DASHBOARD_URL = "http://127.0.0.1:8000/active_alert"

def send_alert_async(alert_type, reason):
    """Send alert to dashboard in a separate thread to avoid blocking video processing"""
    def _send():
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            payload = {
                "type": alert_type,
                "details": {"reason": reason},
                "timestamp": timestamp
            }
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                DASHBOARD_URL, 
                data=data, 
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=1) as response:
                pass # Success
        except Exception as e:
            # Silently fail if dashboard is down to keep CV running
            pass
            
    # Run in thread
    threading.Thread(target=_send, daemon=True).start()

# State variables
look_away_start = None
alert_active = False
alert_start_time = None
last_face_center = None
face_center_history = deque(maxlen=10)
face_angle_history = deque(maxlen=10)
eye_detection_history = deque(maxlen=10)
distraction_confidence = 0.0
no_face_start_time = None
countdown_value = 10

print("Press 'q' to quit the application.")

def is_face_centered(face_x, face_w, frame_width, frame_height, face_y, face_h):
    """Check if face is centered in the frame (both X and Y axes)"""
    face_center_x = face_x + face_w / 2
    face_center_y = face_y + face_h / 2
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    # Calculate normalized deviation for both axes
    x_deviation = abs(face_center_x - frame_center_x) / frame_width
    y_deviation = abs(face_center_y - frame_center_y) / frame_height
    
    # Face is centered if both deviations are within threshold
    return x_deviation < FACE_CENTER_THRESHOLD and y_deviation < FACE_CENTER_THRESHOLD

def detect_eyes_in_face(gray, face_roi):
    """Detect eyes within the face region with improved precision"""
    x, y, w, h = face_roi
    
    # Focus on upper portion of face where eyes are located
    eye_region_y = y
    eye_region_h = int(h * EYE_DETECTION_RATIO)
    eye_region = gray[eye_region_y:eye_region_y+eye_region_h, x:x+w]
    
    # Improved eye detection parameters for better accuracy
    eyes = eye_cascade.detectMultiScale(
        eye_region,
        scaleFactor=1.05,      # Smaller scale factor for more precision
        minNeighbors=4,        # Higher minNeighbors for fewer false positives
        minSize=(max(15, w//10), max(15, h//15)),  # Adaptive min size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Filter and validate detected eyes
    validated_eyes = []
    for (ex, ey, ew, eh) in eyes:
        # Check if eye is in reasonable position (not too low or too high)
        eye_relative_y = ey / eye_region_h if eye_region_h > 0 else 0
        if 0.1 < eye_relative_y < 0.8:  # Eyes should be in middle portion
            # Adjust eye coordinates to full frame
            validated_eyes.append((ex + x, ey + y, ew, eh))
    
    return validated_eyes

def calculate_face_angle_precise(face_x, face_w, frame_width, eyes):
    """Calculate more precise face angle using eye positions and face position"""
    face_center_x = face_x + face_w / 2
    frame_center_x = frame_width / 2
    
    # Method 1: Face position-based angle
    position_offset = (face_center_x - frame_center_x) / frame_width
    position_angle = position_offset * 40  # More conservative multiplier
    
    # Method 2: Eye-based angle (if both eyes detected)
    eye_angle = 0
    if len(eyes) >= 2:
        # Sort eyes by x position (left to right)
        sorted_eyes = sorted(eyes, key=lambda e: e[0])
        left_eye = sorted_eyes[0]
        right_eye = sorted_eyes[1]
        
        # Calculate eye centers
        left_eye_center_x = left_eye[0] + left_eye[2] / 2
        right_eye_center_x = right_eye[0] + right_eye[2] / 2
        
        # Calculate inter-eye distance
        inter_eye_dist = abs(right_eye_center_x - left_eye_center_x)
        
        # Expected inter-eye distance for frontal face (approximately 0.3-0.4 of face width)
        expected_dist = face_w * 0.35
        
        # If inter-eye distance is significantly different, face is turned
        if inter_eye_dist > 0:
            dist_ratio = inter_eye_dist / expected_dist
            # Calculate angle based on eye asymmetry
            eye_center_x = (left_eye_center_x + right_eye_center_x) / 2
            eye_offset = (eye_center_x - face_center_x) / face_w
            eye_angle = eye_offset * 30 + (1 - dist_ratio) * 20
    
    # Combine both methods with weighting
    if len(eyes) >= 2:
        # If we have both eyes, trust eye-based calculation more
        final_angle = 0.3 * position_angle + 0.7 * eye_angle
    else:
        # If only one or no eyes, rely more on position
        final_angle = 0.7 * position_angle + 0.3 * eye_angle
    
    return final_angle

def analyze_eye_gaze(eyes, face_roi):
    """Analyze if eyes are looking forward based on position and symmetry"""
    if len(eyes) < 2:
        return False, "Insufficient eyes"
    
    x, y, w, h = face_roi
    sorted_eyes = sorted(eyes, key=lambda e: e[0])
    left_eye = sorted_eyes[0]
    right_eye = sorted_eyes[1]
    
    # Check eye positions relative to face
    left_eye_center_x = left_eye[0] + left_eye[2] / 2
    right_eye_center_x = right_eye[0] + right_eye[2] / 2
    
    # Eyes should be roughly symmetric around face center
    face_center_x = x + w / 2
    left_dist = abs(left_eye_center_x - (face_center_x - w * 0.15))
    right_dist = abs(right_eye_center_x - (face_center_x + w * 0.15))
    
    # Check if eyes are at appropriate height (upper portion of face)
    eye_y_avg = (left_eye[1] + right_eye[1]) / 2
    expected_eye_y = y + h * 0.25
    eye_y_deviation = abs(eye_y_avg - expected_eye_y) / h
    
    # Eyes are looking forward if symmetric and at correct height
    is_forward = (left_dist + right_dist) / w < 0.3 and eye_y_deviation < 0.2
    
    return is_forward, "Eyes forward" if is_forward else "Eyes not aligned"

def is_face_moving_away(face_center_x, frame_width):
    """Detect if face is moving away from center using improved history analysis"""
    face_center_history.append(face_center_x)
    
    if len(face_center_history) < 5:
        return False
    
    frame_center = frame_width / 2
    recent_positions = list(face_center_history)[-5:]
    
    # Calculate deviations from center
    deviations = [abs(pos - frame_center) for pos in recent_positions]
    
    # Check for consistent movement away from center
    if len(deviations) >= 5:
        # Calculate trend (increasing deviation)
        trend = (deviations[-1] - deviations[0]) / len(deviations)
        
        # Check if deviation is above threshold and increasing
        if deviations[-1] > frame_width * FACE_CENTER_THRESHOLD and trend > 0:
            # Additional check: consistent direction of movement
            positions_diff = [recent_positions[i+1] - recent_positions[i] for i in range(len(recent_positions)-1)]
            consistent_direction = all(d > 0 for d in positions_diff) or all(d < 0 for d in positions_diff)
            
            if consistent_direction:
                return True
    
    return False

def smooth_angle(angle):
    """Apply temporal smoothing to face angle to reduce noise"""
    face_angle_history.append(angle)
    
    if len(face_angle_history) < 3:
        return angle
    
    # Use median filter for robustness against outliers
    recent_angles = list(face_angle_history)[-SMOOTHING_WINDOW:]
    return np.median(recent_angles)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better detection in varying lighting
    gray = cv2.equalizeHist(gray)

    frame_h, frame_w = frame.shape[:2]
    face_detected = False
    distracted = False
    distraction_reason = ""

    # Detect faces with improved parameters for better precision
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,      # Smaller scale for more precise detection
        minNeighbors=7,        # Higher threshold for fewer false positives
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        maxSize=(frame_w, frame_h),  # Limit max size for efficiency
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Sort faces by size (largest first) and process only the largest
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        face_detected = True
        
        # Reset countdown when face is detected
        no_face_start_time = None
        countdown_value = 10

        # Check if face is large enough
        if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
            face_center_x = x + w / 2
            face_center_y = y + h / 2
            
            # Detect eyes in the face region with improved method
            eyes = detect_eyes_in_face(gray, (x, y, w, h))
            
            # Detect mouth/lips in the lower face
            # ROI for mouth (lower 40% of face)
            mouth_roi_y = y + int(h * 0.6)
            mouth_roi_h = int(h * 0.4)
            mouth_roi = gray[mouth_roi_y:mouth_roi_y+mouth_roi_h, x:x+w]
            
            # Detect smiles/mouth movement
            mouths = smile_cascade.detectMultiScale(
                mouth_roi,
                scaleFactor=1.7,
                minNeighbors=20,
                minSize=(25, 25),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # If mouth detected (often triggers on talking/movement)
            lip_movement_detected = len(mouths) > 0
            if lip_movement_detected:
                 # Check previous state to avoid spamming alerts too fast, but we rely on the logic below
                 pass

            # Calculate precise face angle using multiple methods
            raw_face_angle = calculate_face_angle_precise(x, w, frame_w, eyes)
            face_angle = smooth_angle(raw_face_angle)
            
            # Check for distractions with improved methods
            is_centered = is_face_centered(x, w, frame_w, frame_h, y, h)
            is_moving_away = is_face_moving_away(face_center_x, frame_w)
            eyes_detected = len(eyes) >= EYE_DETECTION_CONFIDENCE
            
            # Analyze eye gaze direction
            eyes_forward, gaze_status = analyze_eye_gaze(eyes, (x, y, w, h))
            
            # Update eye detection history for temporal analysis
            eye_detection_history.append(eyes_detected)
            
            # Calculate distraction confidence using multiple factors
            distraction_factors = []
            
            # Factor 1: Face centering
            if not is_centered:
                face_center_x_pos = x + w / 2
                frame_center_x = frame_w / 2
                deviation = abs(face_center_x_pos - frame_center_x) / frame_w
                distraction_factors.append(("Face off-center", min(deviation / FACE_CENTER_THRESHOLD, 1.0)))
            
            # Factor 2: Face angle
            if abs(face_angle) > FACE_ANGLE_THRESHOLD:
                angle_factor = min(abs(face_angle) / 30.0, 1.0)  # Normalize to 0-1
                distraction_factors.append(("Face turned", angle_factor))
            
            # Factor 3: Eye detection
            if not eyes_detected:
                # Check recent history - if eyes missing for multiple frames, higher confidence
                recent_eye_detections = list(eye_detection_history)[-3:]
                eye_missing_ratio = 1.0 - (sum(recent_eye_detections) / len(recent_eye_detections)) if recent_eye_detections else 0
                distraction_factors.append(("Eyes not visible", eye_missing_ratio))
            
            # Factor 4: Eye gaze direction
            if not eyes_forward and len(eyes) >= 2:
                distraction_factors.append(("Eyes not aligned", 0.6))
            
            # Factor 5: Movement away
            if is_moving_away:
                distraction_factors.append(("Moving away", 0.7))
            
            # Factor 6: Lip Movement
            if lip_movement_detected:
                distraction_factors.append(("Lip movement detected", 0.9))
            
            # Calculate overall distraction confidence
            prev_confidence = distraction_confidence  # Store previous value
            if distraction_factors:
                # Weighted average of all factors
                total_weight = sum(weight for _, weight in distraction_factors)
                current_confidence = total_weight / len(distraction_factors)
                
                # Use exponential smoothing for stability (reduce false positives)
                distraction_confidence = 0.7 * current_confidence + 0.3 * prev_confidence
                
                # Determine if distracted based on confidence threshold
                if distraction_confidence >= CONFIDENCE_THRESHOLD:
                    distracted = True
                    # Use the factor with highest weight as reason
                    distraction_reason = max(distraction_factors, key=lambda x: x[1])[0]
                else:
                    distracted = False
                    distraction_reason = ""
            else:
                distracted = False
                distraction_reason = ""
                distraction_confidence = 0.3 * prev_confidence  # Decay when no factors
            
            # Handle alert timing
            if distracted:
                if look_away_start is None:
                    look_away_start = time.time()
                
                # Check duration OR if it's lip movement (trigger faster for talking)
                duration_threshold = LOOK_AWAY_DURATION
                if distraction_reason == "Lip movement detected":
                    duration_threshold = 0.5 # Faster trigger for talking
                
                if time.time() - look_away_start >= duration_threshold:
                    if not alert_active:
                        alert_active = True
                        alert_start_time = time.time()
                        
                        # Determine alert type to send
                        alert_type = "LOOKING_AWAY" # Default
                        if distraction_reason == "Lip movement detected":
                             alert_type = "LIP_MOVEMENT"
                        
                        send_alert_async(alert_type, distraction_reason)
                else:
                    # Still counting down, don't activate alert yet
                    pass
            else:
                # Face is back to normal
                look_away_start = None
                if alert_active:
                    # Keep alert for a moment after returning to normal
                    if alert_start_time and (time.time() - alert_start_time) > 0.5:
                        alert_active = False
                        alert_start_time = None
                else:
                    alert_active = False
                    alert_start_time = None
            
            # Draw face rectangle (red if alert, green otherwise)
            box_color = (0, 0, 255) if alert_active else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
            
            # Draw eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            
            # Draw mouth/lips if detected (Red rectangle if movement detected)
            for (mx, my, mw, mh) in mouths:
                # Adjust coordinates relative to the full frame
                mx += x
                my += mouth_roi_y
                cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
                
                # Add label
                cv2.putText(frame, "Talking/Moving", (mx, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Draw alert message inside the rectangle in small text
            if alert_active:
                alert_duration = int(time.time() - alert_start_time) if alert_start_time else 0
                alert_text = f"ALERT: {distraction_reason} ({alert_duration}s)"
                
                # Small font for text inside rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(alert_text, font, font_scale, thickness)
                
                # Position inside rectangle at bottom-left corner
                text_x = x + 5
                text_y = y + h - 5
                
                # Ensure text stays within rectangle bounds
                if text_x + text_width > x + w:
                    text_x = x + w - text_width - 5
                if text_y - text_height < y:
                    text_y = y + text_height + 5
                
                # Draw text background for better visibility
                cv2.rectangle(frame, 
                            (text_x - 3, text_y - text_height - 3),
                            (text_x + text_width + 3, text_y + baseline + 3),
                            (0, 0, 0), -1)
                
                # Draw alert text in red inside rectangle
                cv2.putText(frame, alert_text, (text_x, text_y), 
                          font, font_scale, (0, 0, 255), thickness)
            
            # Draw face angle for debugging (optional)
            cv2.putText(frame, f"Angle: {face_angle:.1f}°", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Face too small
            face_detected = False
    else:
        # No face detected
        face_detected = False
        look_away_start = None
        if alert_active and alert_start_time:
            # Keep alert for a moment after face disappears
            if (time.time() - alert_start_time) > 0.5:
                alert_active = False
                alert_start_time = None
        else:
            alert_active = False
        face_center_history.clear()
        
        # Initialize countdown timer when face is first not detected
        if no_face_start_time is None:
            no_face_start_time = time.time()
            countdown_value = 10
        
        # Update countdown based on elapsed time
        elapsed_time = time.time() - no_face_start_time
        countdown_value = max(0, int(11 - elapsed_time))
        
        # Send alert if this is a new "no face" event (throttled)
        # We use a simple latch to avoid spamming: sending once per disappearance is usually enough, 
        # but here we'll send it if 'no_face_start_time' was just set (implied by context logic, strictly speaking we'd want a flag)
        # Instead, let's track if we sent it.
        if elapsed_time < 0.2: # Rough check to send only at start
             send_alert_async("FACE_NOT_VISIBLE", "User left the camera view")
        
        # Display "No face detected" message at top left (blinking)
        no_face_text = "No face detected"
        font = cv2.FONT_HERSHEY_SIMPLEX  # Arial style font
        font_scale = 0.48  # Size 12
        thickness = 2
        color = (0, 0, 255)  # Red color in BGR format
        
        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(no_face_text, font, font_scale, thickness)
        
        # Calculate position for top left
        text_x = 20  # Left margin
        text_y = 30  # Top position with margin
        
        # Blinking effect (toggle every 0.5 seconds)
        current_time = time.time()
        blink_on = int(current_time * 2) % 2 == 0
        
        # Draw the text only when blinking is on
        if blink_on:
            cv2.putText(frame, no_face_text, (text_x, text_y), 
                       font, font_scale, color, thickness)
        
        # Display countdown timer at top right
        countdown_text = f"return in: {countdown_value}"
        countdown_font = cv2.FONT_HERSHEY_SIMPLEX  # Arial style font
        countdown_font_scale = 0.48  # Size 12
        countdown_thickness = 2
        countdown_color = (0, 0, 255)  # Red color in BGR format
        
        # Get countdown text size
        (countdown_width, countdown_height), countdown_baseline = cv2.getTextSize(
            countdown_text, countdown_font, countdown_font_scale, countdown_thickness)
        
        # Calculate position for top right
        countdown_x = frame_w - countdown_width - 20  # Right aligned with margin
        countdown_y = 30  # Top position with margin
        
        # Draw the countdown text
        cv2.putText(frame, countdown_text, (countdown_x, countdown_y), 
                   countdown_font, countdown_font_scale, countdown_color, countdown_thickness)

    # Display the frame
    cv2.imshow('DRISHTI', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()