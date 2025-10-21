import os
import cv2
import numpy as np


def setup_detectors():
    """
    Initialize all OpenCV detection utilities.
    Loads Haar cascades for face and eye detection
    and prepares a blob detector for eye tracking.
    """
    face_cascade_path = os.path.join("Classifiers", "haar", "haarcascade_frontalface_default.xml")
    eye_cascade_path = os.path.join("Classifiers", "haar", "haarcascade_eye.xml")

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    blob_params = cv2.SimpleBlobDetector_Params()
    blob_params.filterByArea = True
    blob_params.maxArea = 1500
    blob_detector = cv2.SimpleBlobDetector_create(blob_params)

    return face_cascade, eye_cascade, blob_detector


def find_face(image, gray_image, cascade):
    """
    Identify the primary face in the frame.
    If multiple faces are detected, the largest is selected.

    Returns:
        face_color (np.ndarray): cropped color frame of the face
        face_gray (np.ndarray): cropped grayscale frame
        left_region (tuple): X-range for left eye
        right_region (tuple): X-range for right eye
        x, y (int): coordinates of top-left corner of the detected face
    """
    detections = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    if len(detections) == 0:
        return None, None, None, None, None, None

    # Choose the biggest detected face
    x, y, w, h = max(detections, key=lambda rect: rect[3])
    face_color = image[y:y + h, x:x + w]
    face_gray = gray_image[y:y + h, x:x + w]

    left_region = (int(w * 0.1), int(w * 0.45))
    right_region = (int(w * 0.55), int(w * 0.9))

    return face_color, face_gray, left_region, right_region, x, y


def locate_eyes(frame, gray_frame, left_range, right_range, cascade):
    """
    Detects left and right eyes from a face frame using Haar cascades.

    Returns:
        left_eye, right_eye, left_eye_gray, right_eye_gray
    """
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)
    left_eye = right_eye = left_eye_gray = right_eye_gray = None

    for (x, y, w, h) in eyes:
        eye_center = int(x + w / 2)

        if left_range[0] <= eye_center <= left_range[1]:
            left_eye = frame[y:y + h, x:x + w]
            left_eye_gray = gray_frame[y:y + h, x:x + w]
            left_eye, left_eye_gray = trim_eyebrows(left_eye, left_eye_gray)

        elif right_range[0] <= eye_center <= right_range[1]:
            right_eye = frame[y:y + h, x:x + w]
            right_eye_gray = gray_frame[y:y + h, x:x + w]
            right_eye, right_eye_gray = trim_eyebrows(right_eye, right_eye_gray)

    return left_eye, right_eye, left_eye_gray, right_eye_gray


def refine_eye(img_gray, thresh_value, detector, prev_size=None):
    """
    Process an eye region and detect keypoints (pupil-like blobs).
    """
    _, thresh = cv2.threshold(img_gray, thresh_value, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 5)

    keypoints = detector.detect(thresh)
    if not keypoints:
        return keypoints

    # Optionally filter blobs based on previous frame area
    if prev_size and len(keypoints) > 1:
        keypoints = [min(keypoints, key=lambda k: abs(k.size - prev_size))]

    return keypoints


def trim_eyebrows(img_color, img_gray):
    """Removes upper eyebrow region (15px from top)."""
    h, w = img_color.shape[:2]
    return img_color[15:h, 0:w], img_gray[15:h, 0:w]


def render_keypoints(frame, keypoints):
    """Overlay detected blobs onto the image."""
    cv2.drawKeypoints(frame, keypoints, frame, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
