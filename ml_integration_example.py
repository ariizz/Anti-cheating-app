"""
Integration example for ML-based high-precision detection
This file demonstrates how to use the ML detector with the existing proctoring system
"""

import cv2
from ml_detector import MLDetector
import time

def main():
    """
    Example usage of the ML detector
    """
    # Initialize ML detector
    detector = MLDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("=== ML-Based Proctoring System ===")
    print("Starting initial face scan...")
    print("Please look at the camera for 3 seconds...")
    
    # Scanning phase
    scan_start = time.time()
    scan_complete = False
    
    while not scan_complete:
        ret, frame = cap.read()
        if not ret:
            break
        
        elapsed = time.time() - scan_start
        
        # Display scanning progress
        cv2.putText(frame, f"SCANNING... {int(3 - elapsed)}s", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if elapsed > 3.0:
            if detector.scan_reference_face(frame):
                scan_complete = True
                print("✓ Scan complete! Starting monitoring...")
                cv2.putText(frame, "SCAN COMPLETE!", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('ML Proctoring', frame)
                cv2.waitKey(1000)
            else:
                print("× Scan failed. Please ensure your face is clearly visible.")
                scan_start = time.time()
        
        cv2.imshow('ML Proctoring', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    # Monitoring phase
    print("\n=== Monitoring Active ===")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform comprehensive analysis
        analysis = detector.get_comprehensive_analysis(frame)
        
        # Draw overlay with results
        frame = detector.draw_analysis_overlay(frame, analysis)
        
        # Print alerts
        if analysis["alerts"]:
            for alert_type, alert_msg in analysis["alerts"]:
                print(f"⚠️  [{alert_type}] {alert_msg}")
                
                # Here you would call send_alert_async(alert_type, alert_msg)
                # to integrate with the existing dashboard system
        
        # Display confidence scores
        info_y = frame.shape[0] - 100
        cv2.putText(frame, f"Identity: {analysis['identity_confidence']:.2%}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        cv2.putText(frame, f"Gaze: {analysis['gaze_confidence']:.2%}", 
                   (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        cv2.imshow('ML Proctoring', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Monitoring session ended")

if __name__ == "__main__":
    main()
