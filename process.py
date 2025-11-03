import os
import cv2
import numpy as np


def setup_detectors():
    """
    Initialize all essential OpenCV detection utilities.
    Loads **Haar cascades** for face and eye detection
    and prepares the **SimpleBlobDetector** for robust eye tracking.
    """
    face_cascade_loc = os.path.join("Classifiers", "haar", "haarcascade_frontalface_default.xml")
    eye_cascade_loc = os.path.join("Classifiers", "haar", "haarcascade_eye.xml")

    # Load the detection algorithms
    face_classifier = cv2.CascadeClassifier(face_cascade_loc)
    eye_classifier = cv2.CascadeClassifier(eye_cascade_loc)

    # Setup SimpleBlobDetector parameters
    blob_parameters = cv2.SimpleBlobDetector_Params()
    blob_parameters.filterByArea = True
    blob_parameters.maxArea = 1500  # Max blob area in pixels
    blob_detector_instance = cv2.SimpleBlobDetector_create(blob_parameters)

    return face_classifier, eye_classifier, blob_detector_instance


def find_face(image, gray_image, cascade):
    """
    Identifies the primary face in the current frame.
    Selects the largest face detected if multiple are present.

    Returns:
        face_color_frame (np.ndarray): Cropped color frame of the face region.
        face_gray_frame (np.ndarray): Cropped grayscale frame of the face region.
        left_eye_range (tuple): X-range boundary for left eye search (relative to face crop).
        right_eye_range (tuple): X-range boundary for right eye search.
        x_coord, y_coord (int): Top-left coordinates of the face in the original image.
    """
    face_detections = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    if len(face_detections) == 0:
        return None, None, None, None, None, None

    # Choose the largest detected face (based on height/width)
    x_coord, y_coord, w, h = max(face_detections, key=lambda rect: rect[3])
    face_color_roi = image[y_coord:y_coord + h, x_coord:x_coord + w]
    face_gray_roi = gray_image[y_coord:y_coord + h, x_coord:x_coord + w]

    left_eye_range = (int(w * 0.1), int(w * 0.45))
    right_eye_range = (int(w * 0.55), int(w * 0.9))

    return face_color_roi, face_gray_roi, left_eye_range, right_eye_range, x_coord, y_coord


def locate_eyes(frame, gray_frame, left_range, right_range, cascade):
    """
    Detects left and right eye regions from a facial frame using the eye Haar cascade.

    Returns:
        left_eye_color, right_eye_color, left_eye_gray, right_eye_gray (Numpy arrays or None)
    """
    eye_detections = cascade.detectMultiScale(gray_frame, 1.3, 5)
    left_eye_color = right_eye_color = left_eye_gray = right_eye_gray = None

    for (x, y, w, h) in eye_detections:
        eye_center_x_pos = int(x + w / 2)

        # Check if the eye detection falls within the left half of the face
        if left_range[0] <= eye_center_x_pos <= left_range[1]:
            left_eye_color = frame[y:y + h, x:x + w]
            left_eye_gray = gray_frame[y:y + h, x:x + w]
            left_eye_color, left_eye_gray = trim_eyebrows(left_eye_color, left_eye_gray)

        # Check if the eye detection falls within the right half of the face
        elif right_range[0] <= eye_center_x_pos <= right_range[1]:
            right_eye_color = frame[y:y + h, x:x + w]
            right_eye_gray = gray_frame[y:y + h, x:x + w]
            right_eye_color, right_eye_gray = trim_eyebrows(right_eye_color, right_eye_gray)

    return left_eye_color, right_eye_color, left_eye_gray, right_eye_gray


def refine_eye(img_gray, thresh_val, detector, prev_size_area=None):
    """
    Processes a grayscale eye region and detects pupil keypoints using blob detection.
    """
    _, thresh = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)
    # Apply morphological operations to refine the pupil shape
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 5)

    keypoints_list = detector.detect(thresh)
    if not keypoints_list:
        return keypoints_list

    # Filter blobs based on previous frame area for stability (if previous data exists)
    if prev_size_area and len(keypoints_list) > 1:
        keypoints_list = [min(keypoints_list, key=lambda k: abs(k.size - prev_size_area))]

    return keypoints_list


def trim_eyebrows(img_color, img_gray):
    """Utility to remove the upper eyebrow region (first 15 pixels from the top)."""
    h, w = img_color.shape[:2]
    # Return the cropped image (color and gray)
    return img_color[15:h, 0:w], img_gray[15:h, 0:w]


def render_keypoints(frame, keypoints):
    """Draws the detected keypoints (pupil blobs) onto the frame."""
    cv2.drawKeypoints(frame, keypoints, frame, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
