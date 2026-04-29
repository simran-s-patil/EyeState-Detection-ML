"""
Eye State Inference Script
==========================

This script provides functionality to:
1. Predict eye state for individual images
2. Real-time eye detection using webcam

Usage:
- Single image: python inference.py --image path/to/image.jpg
- Webcam: python inference.py --webcam
"""

import cv2
import numpy as np
import argparse
import sys
from eye_classifier import EyeClassifier


def predict_single_image(image_path, model_path='eye_model.h5'):
    """
    Predict eye state for a single image.

    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the trained model

    Returns:
        tuple: (prediction_label, confidence_score)
    """
    # Initialize classifier and load model
    classifier = EyeClassifier()
    classifier.load_model(model_path)

    try:
        # Make prediction
        label, confidence = classifier.predict_image(image_path)

        # Print results
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")

        return label, confidence

    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None


def detect_eyes_webcam(model_path='eye_model.h5'):
    """
    Real-time eye detection using webcam.

    Args:
        model_path (str): Path to the trained model
    """
    # Initialize classifier and load model
    classifier = EyeClassifier()
    classifier.load_model(model_path)

    # Load Haar cascade for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if eye_cascade.empty():
        print("Error: Could not load Haar cascade for eye detection.")
        print("Make sure OpenCV is properly installed.")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam eye detection...")
    print("Press 'q' to quit.")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in eyes:
            # Extract eye region
            eye_roi = gray[y:y+h, x:x+w]

            try:
                # Resize to model input size
                eye_resized = cv2.resize(eye_roi, (64, 64))
                eye_normalized = eye_resized.astype('float32') / 255.0
                eye_input = np.expand_dims(eye_normalized, axis=[0, -1])

                # Make prediction
                prediction = classifier.model.predict(eye_input, verbose=0)[0][0]
                confidence = max(prediction, 1 - prediction)

                # Determine label and color
                if prediction > 0.5:
                    label = "Open"
                    color = (0, 255, 0)  # Green
                else:
                    label = "Closed"
                    color = (0, 0, 255)  # Red

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception as e:
                print(f"Error processing eye: {e}")
                continue

        # Display frame
        cv2.imshow('Eye State Detection', frame)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")


def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(description='Eye State Classification Inference')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time detection')
    parser.add_argument('--model', type=str, default='eye_model.h5',
                       help='Path to trained model (default: eye_model.h5)')

    args = parser.parse_args()

    if args.image:
        # Single image prediction
        predict_single_image(args.image, args.model)

    elif args.webcam:
        # Webcam detection
        detect_eyes_webcam(args.model)

    else:
        print("Please specify either --image or --webcam")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()