import cv2
import time
import numpy as np
from collections import deque

# Try to import ML detector for high-precision detection
try:
    from ml_detector import MLDetector
    ML_AVAILABLE = True
    print("✓ ML Detector loaded - Using high-precision detection")
except ImportError:
    ML_AVAILABLE = False
    print("⚠ ML Detector not available - Using standard detection")

# Load the pre-trained face and eye detection models (fallback)
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
EYE_DETECTION_RATIO = 0.5    # eyes should be in upper 45% of face ROI
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
    # Check if enough time has passed since last alert of this type
    current_time = time.time()
    if current_time - last_alert_times.get(alert_type, 0) < ALERT_COOLDOWN:
        return  # Skip this alert, too soon since last one
    
    # Update last alert time
    last_alert_times[alert_type] = current_time
    
    def _send():
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            payload = {
                "type": alert_type,
                "details": {"reason": reason},
                "timestamp": timestamp
            }
            
            # Add to local incident log for display
            incident_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": alert_type,
                "details": reason
            })
            
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

# Incident log for display (store last 10 incidents)
incident_log = deque(maxlen=10)

# Alert throttling - track last alert time for each type to prevent duplicates
last_alert_times = {
    "LOOKING_AWAY": 0,
    "LIP_MOVEMENT": 0,
    "FACE_NOT_VISIBLE": 0,
    "WRONG_FACE": 0,
    "MULTIPLE_FACES": 0
}
ALERT_COOLDOWN = 3.0  # Minimum seconds between same alert type

# Initialize ML Detector if available
ml_detector = None
use_ml_detection = False  # Disabled by default - set to True to enable ML mode

# Try to initialize ML detector
if ML_AVAILABLE:
    try:
        ml_detector = MLDetector()
        # Uncomment the line below to enable ML detection (currently has issues)
        # use_ml_detection = True
        print("⚠ ML Detector available but disabled - Using standard detection for stability")
    except Exception as e:
        print(f"⚠ ML Detector initialization failed: {e}")
        use_ml_detection = False

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
        minNeighbors=3,        # Reduced from 4 to increase sensitivity
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

# Global state for scanning (needs to persist across loop iterations if we werent using a loop, but we are)
scan_complete = False
scan_start_time = None
reference_hist = None
scanning_frames = 0

print("Starting main loop...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,       # Slightly faster, less sensitive to scale noise
        minNeighbors=5,        # Reduced from 7 to detect multiple faces more easily
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # --- SCANNING PHASE ---
    if not scan_complete:
        # Draw scanning overlay
        cv2.putText(frame, "SCANNING USER FACE...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (50, 70), (50 + int(scanning_frames * 2), 90), (0, 255, 0), -1)
        
        if len(faces) == 1:
            if scan_start_time is None:
                scan_start_time = time.time()
                
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Use ML detector for scanning if available
            if use_ml_detection and ml_detector:
                # ML-based face scanning
                scanning_frames += 1
                if time.time() - scan_start_time > 3.0:
                    if ml_detector.scan_reference_face(frame):
                        scan_complete = True
                        print("✓ ML Scan Complete - High precision mode active")
                    else:
                        print("⚠ ML scan failed, retrying...")
                        scan_start_time = time.time()
                        scanning_frames = 0
            else:
                # Fallback: Histogram-based scanning
                roi_color = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                
                if reference_hist is None:
                    reference_hist = hist
                else:
                    cv2.accumulateWeighted(hist, reference_hist, 0.1)
                    
                scanning_frames += 1
                
                if time.time() - scan_start_time > 3.0:
                    scan_complete = True
                    print("Scan Complete")
        else:
            cv2.putText(frame, "Please look at camera (1 face)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Reset if face lost during scan
            scan_start_time = time.time()
            scanning_frames = 0
            
        cv2.imshow('DRISHTI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # --- MONITORING PHASE ---
    
    # Show "Scan Complete" briefly
    if time.time() - scan_start_time < 5.0:
        mode_text = "ML MODE" if use_ml_detection else "STANDARD MODE"
        cv2.putText(frame, f"SCAN COMPLETE - {mode_text} ACTIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    face_detected = False
    distracted = False
    distraction_reason = ""
    wrong_face_detected = False
    multi_face_detected = False
    
    # ========================================================================
    # USE ML DETECTOR IF AVAILABLE (HIGH PRECISION MODE)
    # ========================================================================
    if use_ml_detection and ml_detector:
        # Get comprehensive ML analysis
        analysis = ml_detector.get_comprehensive_analysis(frame)
        
        # Process ML results
        num_faces = analysis["num_faces"]
        
        # Handle multiple faces
        if num_faces > 1:
            multi_face_detected = True
            for (fx, fy, fw, fh) in analysis["face_locations"]:
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 3)
            
            if int(time.time() * 4) % 2 == 0:
                cv2.putText(frame, f"{num_faces} FACES DETECTED", (frame_w//2 - 200, frame_h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            send_alert_async("MULTIPLE_FACES", f"{num_faces} people detected")
        
        # Handle single face
        elif num_faces == 1:
            x, y, w, h = analysis["face_locations"][0]
            
            # Check identity
            if not analysis["is_same_person"]:
                wrong_face_detected = True
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                
                if int(time.time() * 4) % 2 == 0:
                    cv2.putText(frame, "WRONG PERSON DETECTED", (frame_w//2 - 250, frame_h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                send_alert_async("WRONG_FACE", f"Identity mismatch (conf: {analysis['identity_confidence']:.2f})")
            
            else:
                # Identity confirmed - check for distractions
                face_detected = True
                distraction_factors = []
                
                # Gaze detection
                if not analysis["gaze_forward"]:
                    distraction_factors.append(("Looking away", 0.8))
                
                # Head pose
                if not analysis["head_centered"]:
                    pitch, yaw, roll = analysis["head_pose"]
                    distraction_factors.append(("Head turned", 0.7))
                
                # Eye closure
                if not analysis["eyes_open"]:
                    distraction_factors.append(("Eyes closed", 0.6))
                
                # Talking detection
                if analysis["is_talking"]:
                    distraction_factors.append(("Lip movement detected", 0.9))
                
                # Determine if distracted
                if distraction_factors:
                    distracted = True
                    distraction_reason = max(distraction_factors, key=lambda x: x[1])[0]
                
                # Alert logic
                if distracted:
                    if look_away_start is None:
                        look_away_start = time.time()
                    
                    threshold = 0.5 if distraction_reason == "Lip movement detected" else LOOK_AWAY_DURATION
                    
                    if time.time() - look_away_start >= threshold:
                        if not alert_active:
                            alert_active = True
                            alert_start_time = time.time()
                            
                            a_type = "LOOKING_AWAY"
                            if distraction_reason == "Lip movement detected":
                                a_type = "LIP_MOVEMENT"
                            elif distraction_reason == "Eyes closed":
                                a_type = "FACE_NOT_VISIBLE"
                            
                            send_alert_async(a_type, distraction_reason)
                else:
                    look_away_start = None
                    if alert_active and (time.time() - alert_start_time) > 0.5:
                        alert_active = False
                
                # Draw visualization
                color = (0, 0, 255) if alert_active else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Show ML confidence scores
                cv2.putText(frame, f"ID: {analysis['identity_confidence']:.2f}", (x, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
                cv2.putText(frame, f"Gaze: {analysis['gaze_direction']}", (x, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
                
                if alert_active:
                    cv2.putText(frame, f"ALERT: {distraction_reason}", (x, y-45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # No face detected
        else:
            face_detected = False
            look_away_start = None
            alert_active = False
            
            if no_face_start_time is None:
                no_face_start_time = time.time()
            
            elapsed = time.time() - no_face_start_time
            if elapsed < 0.2:
                send_alert_async("FACE_NOT_VISIBLE", "No face detected")
            
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(frame, "NO FACE DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            remaining = max(0, int(11 - elapsed))
            cv2.putText(frame, f"Return in: {remaining}", (frame_w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # ========================================================================
    # FALLBACK: STANDARD HAAR CASCADE DETECTION
    # ========================================================================
    else:
        # 1. Multi-Face Detection
        if len(faces) > 1:
            multi_face_detected = True
            alert_text = "TWO FACES DETECTED"
            
            # Draw all faces red
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 3)
                
            # Blinking alert
            if int(time.time() * 4) % 2 == 0:
                cv2.putText(frame, alert_text, (frame_w//2 - 200, frame_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            send_alert_async("MULTIPLE_FACES", "Multiple people detected in frame")
            
        # 2. Single Face Identity Check
        elif len(faces) == 1:
            x, y, w, h = faces[0]
            
            # Identity match using Histogram comparison
            roi_color = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            
            # Compare with reference behavior
            match_score = cv2.compareHist(reference_hist, hist, cv2.HISTCMP_CORREL)
            
            # Identity Threshold
            IDENTITY_THRESHOLD = 0.65
            
            # Visualize Score for debugging
            score_color = (0, 255, 0) if match_score >= IDENTITY_THRESHOLD else (0, 0, 255)
            cv2.putText(frame, f"ID Score: {match_score:.2f}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 2)
            
            # Console debug (throttled)
            if int(time.time() * 10) % 20 == 0:
                 print(f"Match Score: {match_score:.4f} (Threshold: {IDENTITY_THRESHOLD})")
            
            if match_score < IDENTITY_THRESHOLD:
                wrong_face_detected = True
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                
                # Blinking "High alert: Other face detected"
                if int(time.time() * 4) % 2 == 0:
                    cv2.putText(frame, "HIGH ALERT: OTHER FACE DETECTED", (frame_w//2 - 250, frame_h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                if not alert_active:
                    send_alert_async("WRONG_FACE", f"Unrecognized face detected (Score: {match_score:.2f})")
                
            else:
                # IDENTITY CONFIRMED
                face_detected = True
                
                # ... (Rest of original distraction logic)
                
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
                    mouth_roi_y = y + int(h * 0.60)
                    mouth_roi_h = int(h * 0.40)
                    # Ensure ROI is valid
                    if mouth_roi_h > 0:
                        mouth_roi = gray[mouth_roi_y:mouth_roi_y+mouth_roi_h, x:x+w]
                        
                        mouths = smile_cascade.detectMultiScale(
                            mouth_roi,
                            scaleFactor=1.7,
                            minNeighbors=20,
                            minSize=(25, 25),
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        lip_movement_detected = len(mouths) > 0
                    else:
                        lip_movement_detected = False

                    # Calculate precise face angle
                    raw_face_angle = calculate_face_angle_precise(x, w, frame_w, eyes)
                    face_angle = smooth_angle(raw_face_angle)
                    
                    is_centered = is_face_centered(x, w, frame_w, frame_h, y, h)
                    is_moving_away = is_face_moving_away(face_center_x, frame_w)
                    eyes_detected = len(eyes) >= EYE_DETECTION_CONFIDENCE
                    eyes_forward, gaze_status = analyze_eye_gaze(eyes, (x, y, w, h))
                    eye_detection_history.append(eyes_detected)
                    
                    distraction_factors = []
                    if not is_centered:
                        distraction_factors.append(("Face off-center", 0.6))
                    if abs(face_angle) > FACE_ANGLE_THRESHOLD:
                        distraction_factors.append(("Face turned", 0.8))
                    if not eyes_detected:
                        distraction_factors.append(("Eyes not visible", 0.5))
                    if not eyes_forward and len(eyes) >= 2:
                        distraction_factors.append(("Eyes not aligned", 0.6))
                    if is_moving_away:
                        distraction_factors.append(("Moving away", 0.7))
                    if lip_movement_detected:
                        distraction_factors.append(("Lip movement detected", 0.9))
                    
                    # Calculate confidence
                    if distraction_factors:
                        current_confidence = sum(w for _, w in distraction_factors) / len(distraction_factors)
                        distraction_confidence = 0.7 * current_confidence + 0.3 * distraction_confidence
                        
                        if distraction_confidence >= CONFIDENCE_THRESHOLD:
                            distracted = True
                            distraction_reason = max(distraction_factors, key=lambda x: x[1])[0]
                        else:
                            distracted = False
                    else:
                        distracted = False
                        distraction_confidence = 0.3 * distraction_confidence
                    
                    # Alert Logic
                    if distracted:
                        if look_away_start is None:
                            look_away_start = time.time()
                        
                        # Custom threshold for lips
                        threshold = 0.5 if distraction_reason == "Lip movement detected" else LOOK_AWAY_DURATION
                        
                        if time.time() - look_away_start >= threshold:
                            if not alert_active:
                                alert_active = True
                                alert_start_time = time.time()
                                type_ = "LIP_MOVEMENT" if distraction_reason == "Lip movement detected" else "LOOKING_AWAY"
                                send_alert_async(type_, distraction_reason)
                    else:
                        look_away_start = None
                        if alert_active and (time.time() - alert_start_time) > 0.5:
                             alert_active = False

                    # Visuals
                    color = (0, 0, 255) if alert_active else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                        
                    if lip_movement_detected:
                         for (mx, my, mw, mh) in mouths:
                            mx += x
                            my += mouth_roi_y
                            cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                            cv2.putText(frame, "Talking", (mx, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if alert_active:
                        cv2.putText(frame, f"ALERT: {distraction_reason}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 3. No Face Detected
        else: # len(faces) == 0
            face_detected = False
            look_away_start = None
            alert_active = False
            
            # (Preserve existing no face logic simply)
            if no_face_start_time is None:
                no_face_start_time = time.time()
                
            elapsed = time.time() - no_face_start_time
            if elapsed < 0.2:
                 send_alert_async("FACE_NOT_VISIBLE", "No face detected")
                 
            # Blink Text
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(frame, "NO FACE DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            remaining = max(0, int(11 - elapsed))
            cv2.putText(frame, f"Return in: {remaining}", (frame_w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ============================================================================
    # PREMIUM SPLIT-SCREEN LAYOUT WITH PROFESSIONAL DESIGN
    # ============================================================================
    
    # Calculate responsive dimensions
    table_width = 600
    total_width = frame_w + table_width
    display_height = max(frame_h, 720)
    
    # Create canvas with gradient background
    canvas = np.zeros((display_height, total_width, 3), dtype=np.uint8)
    
    # Create gradient background for right panel
    for i in range(display_height):
        gradient_value = int(25 + (i / display_height) * 15)
        canvas[i, frame_w:] = [gradient_value, gradient_value, gradient_value]
    
    # Place camera feed on left
    canvas[0:frame_h, 0:frame_w] = frame
    
    # Add elegant border around camera feed
    cv2.rectangle(canvas, (0, 0), (frame_w-1, frame_h-1), (80, 80, 80), 2)
    
    # Vertical separator with gradient
    for i in range(display_height):
        color_val = int(60 + (i / display_height) * 40)
        cv2.line(canvas, (frame_w, i), (frame_w, i+1), (color_val, color_val, color_val), 3)
    
    # ============================================================================
    # RIGHT PANEL: INCIDENT REPORT DASHBOARD
    # ============================================================================
    
    table_x = frame_w + 20
    
    # === HEADER SECTION ===
    header_bg_height = 100
    cv2.rectangle(canvas, (frame_w, 0), (total_width, header_bg_height), (45, 45, 55), -1)
    
    # Main title with shadow effect
    title_text = "INCIDENT DASHBOARD"
    title_x = table_x + 140
    title_y = 35
    # Shadow
    cv2.putText(canvas, title_text, (title_x + 2, title_y + 2), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)
    # Main text with gradient effect
    cv2.putText(canvas, title_text, (title_x, title_y), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (100, 200, 255), 2)
    
    # Subtitle
    subtitle = "Real-time Monitoring & Analytics"
    cv2.putText(canvas, subtitle, (table_x + 160, title_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Status indicator
    status_x = table_x + 200
    status_y = title_y + 55
    # Pulsing green dot
    pulse = int(time.time() * 3) % 2
    dot_color = (0, 255, 0) if pulse else (0, 200, 0)
    cv2.circle(canvas, (status_x, status_y), 6, dot_color, -1)
    cv2.putText(canvas, "LIVE MONITORING", (status_x + 15, status_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    
    # === STATISTICS CARDS ===
    stats_y = header_bg_height + 20
    card_height = 70
    
    # Count incidents by type
    incident_counts = {
        "LOOKING_AWAY": 0,
        "LIP_MOVEMENT": 0,
        "FACE_NOT_VISIBLE": 0,
        "WRONG_FACE": 0,
        "MULTIPLE_FACES": 0
    }
    for inc in incident_log:
        if inc["type"] in incident_counts:
            incident_counts[inc["type"]] += 1
    
    # Total incidents card
    card_x = table_x
    cv2.rectangle(canvas, (card_x, stats_y), (card_x + 180, stats_y + card_height), 
                  (50, 50, 60), -1)
    cv2.rectangle(canvas, (card_x, stats_y), (card_x + 180, stats_y + card_height), 
                  (100, 200, 255), 2)
    cv2.putText(canvas, "TOTAL INCIDENTS", (card_x + 15, stats_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    cv2.putText(canvas, str(len(incident_log)), (card_x + 70, stats_y + 55), 
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (100, 200, 255), 2)
    
    # Critical alerts card
    card_x2 = table_x + 200
    critical_count = incident_counts["WRONG_FACE"] + incident_counts["MULTIPLE_FACES"]
    cv2.rectangle(canvas, (card_x2, stats_y), (card_x2 + 180, stats_y + card_height), 
                  (50, 50, 60), -1)
    cv2.rectangle(canvas, (card_x2, stats_y), (card_x2 + 180, stats_y + card_height), 
                  (0, 100, 255), 2)
    cv2.putText(canvas, "CRITICAL ALERTS", (card_x2 + 20, stats_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    cv2.putText(canvas, str(critical_count), (card_x2 + 70, stats_y + 55), 
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 100, 255), 2)
    
    # === INCIDENT TABLE ===
    table_start_y = stats_y + card_height + 30
    header_height = 45
    row_height = 50
    
    # Table header with gradient
    header_y1 = table_start_y
    header_y2 = table_start_y + header_height
    for i in range(header_height):
        alpha = i / header_height
        color_val = int(70 - alpha * 20)
        cv2.line(canvas, (table_x - 10, header_y1 + i), 
                (total_width - 20, header_y1 + i), 
                (color_val, color_val, color_val + 10), 1)
    
    # Column positions
    col_time_x = table_x + 5
    col_type_x = table_x + 100
    col_details_x = table_x + 280
    col_status_x = table_x + 500
    
    # Header text with icons
    cv2.putText(canvas, "TIME", (col_time_x, table_start_y + 28), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, "TYPE", (col_type_x, table_start_y + 28), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, "DETAILS", (col_details_x, table_start_y + 28), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, "SEVERITY", (col_status_x, table_start_y + 28), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
    
    # Header bottom line
    cv2.line(canvas, (table_x - 10, header_y2), 
             (total_width - 20, header_y2), (100, 200, 255), 2)
    
    # Display incidents
    current_y = header_y2
    max_rows = min(len(incident_log), 8)
    
    for i, incident in enumerate(reversed(list(incident_log))):
        if i >= max_rows:
            break
            
        row_y = current_y + (i * row_height)
        
        # Alternating row background with hover effect
        if i % 2 == 0:
            cv2.rectangle(canvas, (table_x - 10, row_y), 
                         (total_width - 20, row_y + row_height), 
                         (42, 42, 48), -1)
        else:
            cv2.rectangle(canvas, (table_x - 10, row_y), 
                         (total_width - 20, row_y + row_height), 
                         (38, 38, 44), -1)
        
        # Color coding and severity
        type_info = {
            "LOOKING_AWAY": {
                "color": (0, 165, 255),
                "name": "Looking Away",
                "severity": "MEDIUM",
                "sev_color": (0, 165, 255)
            },
            "LIP_MOVEMENT": {
                "color": (0, 200, 255),
                "name": "Lip Movement",
                "severity": "MEDIUM",
                "sev_color": (0, 200, 255)
            },
            "FACE_NOT_VISIBLE": {
                "color": (0, 100, 255),
                "name": "No Face",
                "severity": "HIGH",
                "sev_color": (0, 100, 255)
            },
            "WRONG_FACE": {
                "color": (0, 0, 255),
                "name": "Wrong Person",
                "severity": "CRITICAL",
                "sev_color": (0, 0, 255)
            },
            "MULTIPLE_FACES": {
                "color": (0, 0, 255),
                "name": "Multiple Faces",
                "severity": "CRITICAL",
                "sev_color": (0, 0, 255)
            }
        }
        
        info = type_info.get(incident["type"], {
            "color": (255, 255, 255),
            "name": incident["type"],
            "severity": "LOW",
            "sev_color": (100, 255, 100)
        })
        
        # Time with icon
        cv2.circle(canvas, (col_time_x - 3, row_y + 25), 3, (100, 200, 255), -1)
        cv2.putText(canvas, incident["time"], (col_time_x + 8, row_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        # Type with colored badge
        badge_x = col_type_x - 5
        badge_y = row_y + 15
        cv2.rectangle(canvas, (badge_x, badge_y), 
                     (badge_x + 150, badge_y + 25), 
                     info["color"], -1)
        cv2.rectangle(canvas, (badge_x, badge_y), 
                     (badge_x + 150, badge_y + 25), 
                     info["color"], 1)
        cv2.putText(canvas, info["name"], (badge_x + 8, badge_y + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Details
        details = incident["details"]
        if len(details) > 22:
            details = details[:22] + "..."
        cv2.putText(canvas, details, (col_details_x, row_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        
        # Severity indicator
        sev_x = col_status_x
        sev_y = row_y + 15
        cv2.rectangle(canvas, (sev_x, sev_y), (sev_x + 70, sev_y + 20), 
                     info["sev_color"], 1)
        cv2.putText(canvas, info["severity"], (sev_x + 5, sev_y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, info["sev_color"], 1)
        
        # Row separator
        cv2.line(canvas, (table_x - 10, row_y + row_height), 
                (total_width - 20, row_y + row_height), (60, 60, 60), 1)
    
    # === FOOTER SECTION ===
    footer_y = display_height - 50
    cv2.rectangle(canvas, (frame_w, footer_y), (total_width, display_height), 
                  (35, 35, 45), -1)
    
    # Footer stats
    footer_text = f"Monitoring Session Active | Total: {len(incident_log)} | Critical: {critical_count}"
    cv2.putText(canvas, footer_text, (table_x + 80, footer_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    
    # Timestamp
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(canvas, current_time, (total_width - 100, footer_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

    cv2.imshow('DRISHTI - AI Proctoring System', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()