"""
Demo: Adaptive ML Proctoring System with Continuous Learning
Run this to see the advanced ML system in action
"""

import cv2
import time
from adaptive_ml_proctor import AdaptiveMLProctor

def main():
    print("=" * 60)
    print("ADAPTIVE ML PROCTORING SYSTEM")
    print("Features: Continuous Learning | Anomaly Detection | Auto-Calibration")
    print("=" * 60)
    
    # Initialize adaptive ML proctor
    try:
        proctor = AdaptiveMLProctor()
    except Exception as e:
        print(f"❌ Failed to initialize ML system: {e}")
        print("Please install required libraries:")
        print("pip install -r requirements_ml.txt")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n📸 Starting calibration phase...")
    print("Please look at the camera for 3 seconds...")
    
    # Calibration phase
    calibration_start = time.time()
    calibrated = False
    
    while not calibrated:
        ret, frame = cap.read()
        if not ret:
            break
        
        elapsed = time.time() - calibration_start
        
        # Display calibration progress
        cv2.putText(frame, f"CALIBRATING... {int(3 - elapsed)}s", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if elapsed > 3.0:
            if proctor.scan_and_calibrate(frame):
                calibrated = True
                print("✓ Calibration complete!")
                cv2.putText(frame, "CALIBRATION COMPLETE!", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Adaptive ML Proctor', frame)
                cv2.waitKey(1000)
            else:
                print("⚠ Calibration failed. Retrying...")
                calibration_start = time.time()
        
        cv2.imshow('Adaptive ML Proctor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    # Monitoring phase
    print("\n🎯 Monitoring active - System is learning your behavior patterns")
    print("Press 'q' to quit | Press 's' to save models\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Analyze frame
        analysis = proctor.analyze_frame(frame)
        
        # Draw face boxes
        for (x, y, w, h) in analysis['face_locations']:
            color = (0, 255, 0) if analysis['is_same_person'] else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Display status
        h, w = frame.shape[:2]
        
        # Header
        cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)
        cv2.putText(frame, "ADAPTIVE ML PROCTORING", (10, 30), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 200, 255), 2)
        
        # Stats
        status_text = f"Samples: {proctor.samples_collected} | "
        status_text += f"Model: {'Trained' if proctor.behavior_model else 'Learning'}"
        cv2.putText(frame, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Confidence scores
        y_offset = 100
        cv2.putText(frame, f"Identity: {analysis['confidence']:.2%}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        if analysis['anomaly_score'] > 0:
            cv2.putText(frame, f"Anomaly: {analysis['anomaly_score']:.2%}", 
                       (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 165, 255) if analysis['anomaly_score'] < 0.5 else (0, 0, 255), 1)
        
        # Status indicators
        indicators = [
            ("Gaze", analysis['gaze_forward']),
            ("Eyes", analysis['eyes_detected']),
            ("Quiet", not analysis['is_talking']),
            ("Identity", analysis['is_same_person'])
        ]
        
        for i, (label, status) in enumerate(indicators):
            color = (0, 255, 0) if status else (0, 0, 255)
            cv2.putText(frame, f"{label}: {'✓' if status else '✗'}", 
                       (w - 150, 100 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display alerts
        if analysis['alerts']:
            alert_y = h - 100
            for alert_type, alert_msg in analysis['alerts'][:3]:  # Show max 3
                cv2.putText(frame, f"⚠ {alert_msg}", 
                           (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
                alert_y += 25
                print(f"[{alert_type}] {alert_msg}")
        
        # Learning indicator
        if frame_count % 50 == 0:
            cv2.putText(frame, "Learning...", (w - 120, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        cv2.imshow('Adaptive ML Proctor', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            proctor.save_models()
            print("💾 Models saved!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final save
    proctor.save_models()
    
    print("\n" + "=" * 60)
    print(f"✓ Session complete!")
    print(f"📊 Total samples collected: {proctor.samples_collected}")
    print(f"🧠 Behavior model: {'Trained' if proctor.behavior_model else 'Not enough data'}")
    print(f"💾 Models saved to: {proctor.model_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
