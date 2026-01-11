"""
Advanced ML-based Detection Module for High-Precision Proctoring
Uses deep learning models for superior accuracy in:
- Face recognition and verification
- Gaze estimation
- Head pose estimation
- Facial landmark detection
"""

import cv2
import numpy as np
import mediapipe as mp
# Robust import for mediapipe solutions
try:
    import mediapipe.solutions.face_mesh as mp_face_mesh
    import mediapipe.solutions.face_detection as mp_face_detection
except (ImportError, AttributeError):
    try:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        import mediapipe.python.solutions.face_detection as mp_face_detection
    except (ImportError, AttributeError):
        # Fallback to direct solutions import if possible
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
        from mediapipe.python.solutions import face_detection as mp_face_detection

from scipy.spatial import distance as dist
import time

class MLDetector:
    """High-precision ML-based detector using MediaPipe and advanced algorithms"""
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh for high-precision facial landmarks
        self.mp_face_mesh = mp_face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp_face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full range model
            min_detection_confidence=0.7
        )
        
        # Reference face encoding (set during initial scan)
        self.reference_face_encoding = None
        self.reference_landmarks = None
        
        # Thresholds
        self.GAZE_THRESHOLD = 0.15
        self.HEAD_POSE_THRESHOLD = 60  # degrees - very low sensitivity (only extreme turns)
        self.FACE_SIMILARITY_THRESHOLD = 0.85
        self.EAR_THRESHOLD = 0.25  # Eye Aspect Ratio for blink/closed eyes
        self.MAR_THRESHOLD = 0.28   # Lowered threshold for higher sensitivity to subtle talking
        
    def scan_reference_face(self, frame):
        """
        Scan and store reference face during initial setup
        Returns: True if successful, False otherwise
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks and len(results.multi_face_landmarks) == 1:
            self.reference_landmarks = results.multi_face_landmarks[0]
            # Extract face encoding (simplified - using key landmark positions)
            self.reference_face_encoding = self._extract_face_encoding(
                self.reference_landmarks, frame.shape
            )
            return True
        return False
    
    def _extract_face_encoding(self, landmarks, frame_shape):
        """Extract face encoding from landmarks (Higher Precision)"""
        h, w = frame_shape[:2]
        # Use more facial landmarks for a more unique encoding
        # Key points around eyes, nose, mouth and face contour
        key_points = [
            1, 33, 263, 61, 291, 199, # Center points
            10, 152, 234, 454,        # Top, bottom, left, right
            133, 362, 168, 6, 197     # Refined nose/eye points
        ]
        encoding = []
        for idx in key_points:
            lm = landmarks.landmark[idx]
            encoding.extend([lm.x, lm.y, lm.z])
        return np.array(encoding)
    
    def detect_multiple_faces(self, frame):
        """
        Detect if multiple faces are present
        Returns: (num_faces, face_locations)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            face_locations = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                face_locations.append((x, y, width, height))
            return len(results.detections), face_locations
        return 0, []
    
    def verify_face_identity(self, frame):
        """
        Verify if the detected face matches the reference face
        Returns: (is_same_person, confidence_score)
        """
        if self.reference_face_encoding is None:
            return True, 1.0  # No reference, assume valid
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            current_encoding = self._extract_face_encoding(
                results.multi_face_landmarks[0], frame.shape
            )
            
            # Calculate similarity using cosine similarity
            similarity = self._cosine_similarity(
                self.reference_face_encoding, current_encoding
            )
            
            is_same = similarity >= self.FACE_SIMILARITY_THRESHOLD
            return is_same, similarity
        
        return False, 0.0
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def estimate_gaze_direction(self, frame):
        """
        Estimate gaze direction using iris landmarks
        Returns: (is_looking_forward, gaze_direction, confidence)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Get iris landmarks (MediaPipe provides refined iris landmarks)
            # Left iris: 468-473, Right iris: 473-478
            left_iris = [landmarks.landmark[i] for i in range(468, 473)]
            right_iris = [landmarks.landmark[i] for i in range(473, 478)]
            
            # Get eye corners for reference
            left_eye_left = landmarks.landmark[33]
            left_eye_right = landmarks.landmark[133]
            right_eye_left = landmarks.landmark[362]
            right_eye_right = landmarks.landmark[263]
            
            # Calculate iris position relative to eye corners
            left_iris_center = np.mean([[p.x, p.y] for p in left_iris], axis=0)
            right_iris_center = np.mean([[p.x, p.y] for p in right_iris], axis=0)
            
            left_eye_center = [(left_eye_left.x + left_eye_right.x) / 2,
                              (left_eye_left.y + left_eye_right.y) / 2]
            right_eye_center = [(right_eye_left.x + right_eye_right.x) / 2,
                               (right_eye_left.y + right_eye_right.y) / 2]
            
            # Calculate deviation
            left_deviation = abs(left_iris_center[0] - left_eye_center[0])
            right_deviation = abs(right_iris_center[0] - right_eye_center[0])
            avg_deviation = (left_deviation + right_deviation) / 2
            
            # Determine direction with slight hysteresis for smoothing
            if left_iris_center[0] < left_eye_center[0] - (self.GAZE_THRESHOLD * 1.5):
                direction = "LEFT"
            elif left_iris_center[0] > left_eye_center[0] + (self.GAZE_THRESHOLD * 1.5):
                direction = "RIGHT"
            else:
                direction = "FORWARD"
            
            # Use clarified direction for forward check
            is_forward = direction == "FORWARD"
            
            confidence = 1.0 - min(avg_deviation / self.GAZE_THRESHOLD, 1.0)
            return is_forward, direction, confidence
        
        return False, "UNKNOWN", 0.0
    
    def estimate_head_pose(self, frame):
        """
        Estimate head pose (pitch, yaw, roll)
        Returns: (pitch, yaw, roll, is_centered)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Get key 3D points
            nose_tip = landmarks.landmark[1]
            chin = landmarks.landmark[152]
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]
            left_mouth = landmarks.landmark[61]
            right_mouth = landmarks.landmark[291]
            
            # Convert to image coordinates
            points_2d = np.array([
                [nose_tip.x * w, nose_tip.y * h],
                [chin.x * w, chin.y * h],
                [left_eye.x * w, left_eye.y * h],
                [right_eye.x * w, right_eye.y * h],
                [left_mouth.x * w, left_mouth.y * h],
                [right_mouth.x * w, right_mouth.y * h]
            ], dtype=np.float64)
            
            # 3D model points
            points_3d = np.array([
                [0.0, 0.0, 0.0],          # Nose tip
                [0.0, -330.0, -65.0],     # Chin
                [-225.0, 170.0, -135.0],  # Left eye
                [225.0, 170.0, -135.0],   # Right eye
                [-150.0, -150.0, -125.0], # Left mouth
                [150.0, -150.0, -125.0]   # Right mouth
            ], dtype=np.float64)
            
            # Camera matrix
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                points_3d, points_2d, camera_matrix, dist_coeffs
            )
            
            if success:
                # Convert rotation vector to Euler angles
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                pose_mat = cv2.hconcat((rotation_mat, translation_vec))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
                
                pitch = euler_angles[0][0]
                yaw = euler_angles[1][0]
                roll = euler_angles[2][0]
                
                # Check if head is centered
                is_centered = (abs(pitch) < self.HEAD_POSE_THRESHOLD and 
                             abs(yaw) < self.HEAD_POSE_THRESHOLD)
                
                return pitch, yaw, roll, is_centered
        
        return 0, 0, 0, False
    
    def detect_eye_closure(self, frame):
        """
        Detect if eyes are closed using Eye Aspect Ratio (EAR)
        Returns: (eyes_open, ear_score, left_eye_box, right_eye_box)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Left eye landmarks
            left_indices = [33, 160, 158, 133, 153, 144]
            left_eye = [landmarks.landmark[i] for i in left_indices]
            # Right eye landmarks
            right_indices = [362, 385, 387, 263, 373, 380]
            right_eye = [landmarks.landmark[i] for i in right_indices]
            
            def get_box(eye_lms):
                xs = [lm.x * w for lm in eye_lms]
                ys = [lm.y * h for lm in eye_lms]
                # Add some padding to the eye box
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                pad_w = (x_max - x_min) * 0.2
                pad_h = (y_max - y_min) * 0.5
                return (int(x_min - pad_w), int(y_min - pad_h), int((x_max - x_min) + 2*pad_w), int((y_max - y_min) + 2*pad_h))
            
            left_box = get_box(left_eye)
            right_box = get_box(right_eye)
            
            left_ear = self._calculate_ear(left_eye)
            right_ear = self._calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2
            
            eyes_open = avg_ear > self.EAR_THRESHOLD
            return eyes_open, avg_ear, left_box, right_box
        
        return True, 1.0, None, None
    
    def _calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        # Convert landmarks to 2D points
        points = np.array([[lm.x, lm.y] for lm in eye_landmarks])
        
        # Compute distances
        A = dist.euclidean(points[1], points[5])
        B = dist.euclidean(points[2], points[4])
        C = dist.euclidean(points[0], points[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_mouth_movement(self, frame):
        """
        Detect mouth movement/talking using Mouth Aspect Ratio (MAR)
        Returns: (is_talking, mar_score)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Key Mouth landmarks for robust MAR:
            # 61, 291: Corners (Horizontal)
            # 13, 14: Upper/Lower Lip Center (Vertical)
            p61 = landmarks.landmark[61]
            p291 = landmarks.landmark[291]
            p13 = landmarks.landmark[13]
            p14 = landmarks.landmark[14]
            
            # Calculate MAR: Vertical distance / Horizontal distance
            vertical_dist = dist.euclidean([p13.x, p13.y], [p14.x, p14.y])
            horizontal_dist = dist.euclidean([p61.x, p61.y], [p291.x, p291.y])
            
            if horizontal_dist == 0: return False, 0.0
            
            mar = vertical_dist / horizontal_dist
            
            # Sensitivity threshold: 
            # Closed mouth is usually < 0.1, Talking is > 0.3
            is_talking = mar > self.MAR_THRESHOLD
            
            return is_talking, mar
        
        return False, 0.0
    
    def get_comprehensive_analysis(self, frame):
        """
        Perform comprehensive analysis of the frame
        Returns: dict with all detection results
        """
        num_faces, face_locations = self.detect_multiple_faces(frame)
        
        analysis = {
            "num_faces": num_faces,
            "face_locations": face_locations,
            "is_same_person": True,
            "identity_confidence": 1.0,
            "gaze_forward": True,
            "gaze_direction": "FORWARD",
            "gaze_confidence": 1.0,
            "head_centered": True,
            "head_pose": (0, 0, 0),
            "eyes_open": True,
            "eye_score": 1.0,
            "eye_locations": [],
            "is_talking": False,
            "mouth_score": 0.0,
            "alerts": []
        }
        
        if num_faces == 0:
            analysis["alerts"].append(("FACE_NOT_VISIBLE", "No face detected"))
            return analysis
        
        if num_faces > 1:
            analysis["alerts"].append(("MULTIPLE_FACES", f"{num_faces} faces detected"))
            return analysis
        
        # Single face - perform detailed analysis
        is_same, identity_conf = self.verify_face_identity(frame)
        analysis["is_same_person"] = is_same
        analysis["identity_confidence"] = identity_conf
        
        if not is_same:
            analysis["alerts"].append(("WRONG_FACE", f"Different person detected (confidence: {identity_conf:.2f})"))
        
        # Gaze detection
        gaze_forward, gaze_dir, gaze_conf = self.estimate_gaze_direction(frame)
        analysis["gaze_forward"] = gaze_forward
        analysis["gaze_direction"] = gaze_dir
        analysis["gaze_confidence"] = gaze_conf
        
        if not gaze_forward:
            analysis["alerts"].append(("LOOKING_AWAY", f"Gaze direction: {gaze_dir}"))
        
        # Head pose
        pitch, yaw, roll, head_centered = self.estimate_head_pose(frame)
        analysis["head_centered"] = head_centered
        analysis["head_pose"] = (pitch, yaw, roll)
        
        if not head_centered:
            analysis["alerts"].append(("HEAD_TURNED", f"Head pose: pitch={pitch:.1f}°, yaw={yaw:.1f}°"))
        
        # Eye closure
        eyes_open, eye_score, l_box, r_box = self.detect_eye_closure(frame)
        analysis["eyes_open"] = eyes_open
        analysis["eye_score"] = eye_score
        if l_box and r_box:
            analysis["eye_locations"] = [l_box, r_box]
        
        if not eyes_open:
            analysis["alerts"].append(("EYES_CLOSED", f"Eye aspect ratio: {eye_score:.2f}"))
        
        # Mouth movement
        is_talking, mouth_score = self.detect_mouth_movement(frame)
        analysis["is_talking"] = is_talking
        analysis["mouth_score"] = mouth_score
        
        if is_talking:
            analysis["alerts"].append(("LIP_MOVEMENT", f"Mouth aspect ratio: {mouth_score:.2f}"))
        
        return analysis
    
    def draw_analysis_overlay(self, frame, analysis):
        """Draw analysis results on frame"""
        h, w = frame.shape[:2]
        
        # Draw face boxes
        for (x, y, fw, fh) in analysis["face_locations"]:
            color = (0, 255, 0) if analysis["is_same_person"] else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)
        
        # Draw status text
        y_offset = 30
        status_texts = [
            f"Faces: {analysis['num_faces']}",
            f"Identity: {analysis['identity_confidence']:.2f}",
            f"Gaze: {analysis['gaze_direction']}",
            f"Head: {'Centered' if analysis['head_centered'] else 'Turned'}",
            f"Eyes: {'Open' if analysis['eyes_open'] else 'Closed'}",
            f"Talking: {'Yes' if analysis['is_talking'] else 'No'}"
        ]
        
        for text in status_texts:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 25
        
        return frame
