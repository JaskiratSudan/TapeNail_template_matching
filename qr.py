import cv2
import numpy as np

def detect_qr_code():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize QR Code detector
    qr_detector = cv2.QRCodeDetector()

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        # Detect and decode QR codes
        retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)

        if retval:
            for qr_info, qr_points in zip(decoded_info, points):
                if qr_info:
                    # Convert points to integers
                    qr_points = qr_points.astype(int)
                    
                    # Draw polygon around the QR code
                    cv2.polylines(frame, [qr_points], True, (0, 255, 0), 2)
                    
                    # Add text with QR code data
                    cv2.putText(frame, qr_info, tuple(qr_points[0]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('QR Code Scanner', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_qr_code()