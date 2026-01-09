"""
Advanced Adaptive ML Proctoring System (Compatible Version)
Uses scikit-learn for continuous learning without MediaPipe dependency
"""

import cv2
import numpy as np
import pickle
import os
from datetime import datetime
from collections import deque
import json

# ML Libraries (all compatible with Python 3.9)
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠ Install scikit-learn: pip install scikit-learn joblib")


class AdaptiveMLProctor:
    """
    Adaptive ML-based proctoring with continuous learning
    Uses OpenCV + scikit-learn for compatibility
    """
    
    def __init__(self, model_dir="ml_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # ML components
        self.behavior_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)  # Dimensionality reduction
        
        # Reference data
        self.reference_features = None
        self.reference_histogram = None
        
        # Training data
        self.training_samples = []
        self.samples_collected = 0
        self.min_samples_for_training = 50
        
        # Adaptive thresholds
        self.thresholds = {
            'face_similarity': 0.65,
            'eye_detection': 2,
            'gaze_angle': 15,
            'mouth_activity': 0.3
        }
        
        # Load existing models
        self.load_models()
        
        print("✓ Adaptive ML Proctor initialized (Compatible Mode)")
    
    def extract_advanced_features(self, frame, face_rect):
        """
        Extract comprehensive features from face region
        Returns high-dimensional feature vector
        """
        x, y, w, h = face_rect
        face_roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # 1. Histogram features (color distribution)
        hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_roi], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv_roi], [1], None, [32], [0, 256])
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        features.extend(hist_h.flatten())
        features.extend(hist_s.flatten())
        
        # 2. Texture features (LBP-like)
        resized = cv2.resize(gray_roi, (64, 64))
        features.extend(resized.flatten()[::16])  # Downsample for speed
        
        # 3. Edge features
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        features.append(edge_density)
        
        # 4. Geometric features
        features.extend([x/frame.shape[1], y/frame.shape[0], w/frame.shape[1], h/frame.shape[0]])
        
        # 5. Eye detection features
        eyes = self.eye_cascade.detectMultiScale(gray_roi, 1.1, 5)
        features.append(len(eyes))
        if len(eyes) >= 2:
            # Eye positions
            eye_y_avg = np.mean([ey for (ex, ey, ew, eh) in eyes])
            features.append(eye_y_avg / h)
        else:
            features.append(0.5)
        
        # 6. Mouth detection features
        mouth_roi = gray_roi[int(h*0.6):, :]
        mouths = self.smile_cascade.detectMultiScale(mouth_roi, 1.7, 20)
        features.append(len(mouths))
        
        # 7. Brightness and contrast
        features.append(np.mean(gray_roi) / 255.0)
        features.append(np.std(gray_roi) / 255.0)
        
        return np.array(features)
    
    def scan_and_calibrate(self, frame):
        """Initial calibration - learn reference features"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 1:
            x, y, w, h = faces[0]
            
            # Extract reference features
            self.reference_features = self.extract_advanced_features(frame, faces[0])
            
            # Store histogram for identity verification
            face_roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            self.reference_histogram = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(self.reference_histogram, self.reference_histogram, 0, 255, cv2.NORM_MINMAX)
            
            print("✓ Calibration complete - Baseline features captured")
            return True
        return False
    
    def analyze_frame(self, frame):
        """Comprehensive frame analysis with ML"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        
        analysis = {
            'num_faces': len(faces),
            'face_locations': [],
            'is_same_person': True,
            'confidence': 1.0,
            'gaze_forward': True,
            'eyes_detected': False,
            'is_talking': False,
            'anomaly_score': 0.0,
            'alerts': [],
            'features': None
        }
        
        # Multiple faces
        if len(faces) > 1:
            analysis['face_locations'] = [(x, y, w, h) for (x, y, w, h) in faces]
            analysis['alerts'].append(('MULTIPLE_FACES', f"{len(faces)} faces detected"))
            return analysis
        
        # No face
        if len(faces) == 0:
            analysis['alerts'].append(('FACE_NOT_VISIBLE', 'No face detected'))
            return analysis
        
        # Single face analysis
        x, y, w, h = faces[0]
        analysis['face_locations'] = [(x, y, w, h)]
        
        # Extract current features
        current_features = self.extract_advanced_features(frame, faces[0])
        analysis['features'] = current_features
        
        # Identity verification
        if self.reference_features is not None:
            # Use histogram comparison for identity
            face_roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            current_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(current_hist, current_hist, 0, 255, cv2.NORM_MINMAX)
            
            similarity = cv2.compareHist(self.reference_histogram, current_hist, cv2.HISTCMP_CORREL)
            analysis['confidence'] = similarity
            analysis['is_same_person'] = similarity >= self.thresholds['face_similarity']
            
            if not analysis['is_same_person']:
                analysis['alerts'].append(('WRONG_FACE', f'Identity mismatch (conf: {similarity:.2f})'))
                return analysis
        
        # Eye detection
        face_roi_gray = gray[y:y+h, x:x+w]
        eye_region = face_roi_gray[0:int(h*0.5), :]
        eyes = self.eye_cascade.detectMultiScale(eye_region, 1.1, 5)
        analysis['eyes_detected'] = len(eyes) >= self.thresholds['eye_detection']
        
        if not analysis['eyes_detected']:
            analysis['alerts'].append(('EYES_NOT_VISIBLE', 'Eyes not detected'))
        
        # Gaze estimation (simple - based on face position)
        frame_center_x = frame.shape[1] / 2
        face_center_x = x + w / 2
        deviation = abs(face_center_x - frame_center_x) / frame.shape[1]
        analysis['gaze_forward'] = deviation < 0.15
        
        if not analysis['gaze_forward']:
            analysis['alerts'].append(('LOOKING_AWAY', f'Face off-center: {deviation:.2f}'))
        
        # Mouth activity detection
        mouth_roi = face_roi_gray[int(h*0.6):, :]
        if mouth_roi.shape[0] > 0:
            mouths = self.smile_cascade.detectMultiScale(mouth_roi, 1.7, 20)
            analysis['is_talking'] = len(mouths) > 0
            
            if analysis['is_talking']:
                analysis['alerts'].append(('LIP_MOVEMENT', 'Mouth movement detected'))
        
        # Anomaly detection (if model trained)
        if self.behavior_model is not None and ML_AVAILABLE:
            try:
                # Prepare features
                features_scaled = self.scaler.transform(current_features.reshape(1, -1))
                features_pca = self.pca.transform(features_scaled)
                
                # Predict anomaly
                prediction = self.behavior_model.predict(features_pca)[0]
                score = self.behavior_model.score_samples(features_pca)[0]
                
                # Normalize score to 0-1
                anomaly_score = 1.0 / (1.0 + np.exp(score))
                analysis['anomaly_score'] = anomaly_score
                
                if anomaly_score > 0.6:
                    analysis['alerts'].append(('ANOMALY', f'Unusual behavior (score: {anomaly_score:.2f})'))
            except Exception as e:
                pass
        
        # Collect training data (normal behavior only)
        if len(analysis['alerts']) == 0 and ML_AVAILABLE:
            self._collect_training_sample(current_features)
        
        return analysis
    
    def _collect_training_sample(self, features):
        """Collect samples for continuous learning"""
        self.training_samples.append(features)
        self.samples_collected += 1
        
        # Train model periodically
        if self.samples_collected >= self.min_samples_for_training and self.samples_collected % 25 == 0:
            self._train_behavior_model()
    
    def _train_behavior_model(self):
        """Train anomaly detection model"""
        if not ML_AVAILABLE or len(self.training_samples) < self.min_samples_for_training:
            return
        
        print(f"🔄 Training with {len(self.training_samples)} samples...")
        
        try:
            # Prepare data
            X = np.array(self.training_samples)
            
            # Fit scaler and PCA
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled)
            
            # Train Isolation Forest
            self.behavior_model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.behavior_model.fit(X_pca)
            
            # Save models
            self.save_models()
            
            print(f"✓ Model trained successfully!")
        except Exception as e:
            print(f"⚠ Training failed: {e}")
    
    def save_models(self):
        """Save trained models"""
        try:
            if self.behavior_model is not None:
                joblib.dump(self.behavior_model, os.path.join(self.model_dir, 'behavior_model.pkl'))
            
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            joblib.dump(self.pca, os.path.join(self.model_dir, 'pca.pkl'))
            
            calibration_data = {
                'reference_features': self.reference_features,
                'reference_histogram': self.reference_histogram,
                'thresholds': self.thresholds
            }
            with open(os.path.join(self.model_dir, 'calibration.pkl'), 'wb') as f:
                pickle.dump(calibration_data, f)
            
            print(f"💾 Models saved to {self.model_dir}")
        except Exception as e:
            print(f"⚠ Save failed: {e}")
    
    def load_models(self):
        """Load previously trained models"""
        try:
            model_path = os.path.join(self.model_dir, 'behavior_model.pkl')
            if os.path.exists(model_path):
                self.behavior_model = joblib.load(model_path)
                print("✓ Loaded existing behavior model")
            
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            pca_path = os.path.join(self.model_dir, 'pca.pkl')
            if os.path.exists(pca_path):
                self.pca = joblib.load(pca_path)
            
            calibration_path = os.path.join(self.model_dir, 'calibration.pkl')
            if os.path.exists(calibration_path):
                with open(calibration_path, 'rb') as f:
                    data = pickle.load(f)
                    self.reference_features = data.get('reference_features')
                    self.reference_histogram = data.get('reference_histogram')
                    self.thresholds = data.get('thresholds', self.thresholds)
                print("✓ Loaded calibration data")
        except Exception as e:
            print(f"⚠ Load failed: {e}")
