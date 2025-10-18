import sys
import logging
import cv2
import numpy as np
import process  # Assuming this module contains the CV processing functions

# --- PyQt5 Imports ---
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QCloseEvent
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QCheckBox, QSlider
from PyQt5.uic import loadUi

# --- Typing Imports ---
from typing import Optional, Dict, Any, Tuple
from numpy.typing import NDArray

# --- Constants ---
FPS = 30
TIMER_INTERVAL = 1000 // FPS

# Constants for dictionary keys and window identifiers
MAIN_WINDOW_KEY = 'main'
LEFT_EYE_KEY = 'left'
RIGHT_EYE_KEY = 'right'


class EyeTrackerWindow(QMainWindow):
    """Main application window for the Eye Tracker."""

    def __init__(self) -> None:
        """Initializes the main window, UI elements, and CV components."""
        super().__init__()
        loadUi('GUImain.ui', self)
        self._load_stylesheet()

        # --- OpenCV/Camera State ---
        self.capture: Optional[cv2.VideoCapture] = None
        self.camera_is_running: bool = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # --- CV Detectors ---
        # These are expected to be returned by process.init_cv()
        self.face_detector: Optional[cv2.CascadeClassifier] = None
        self.eye_detector: Optional[cv2.CascadeClassifier] = None
        self.blob_detector: Optional[cv2.SimpleBlobDetector] = None
        self._initialize_detectors()

        # --- State Tracking ---
        self.tracking_state: Dict[str, Dict[str, Any]] = {
            LEFT_EYE_KEY: {'keypoints': None, 'area': None},
            RIGHT_EYE_KEY: {'keypoints': None, 'area': None}
        }

        # --- UI Element Mapping ---
        # Assuming these names match the .ui file
        self.image_labels: Dict[str, QLabel] = {
            MAIN_WINDOW_KEY: self.baseImage,
            LEFT_EYE_KEY: self.leftEyeBox,
            RIGHT_EYE_KEY: self.rightEyeBox
        }

        self._connect_signals()

    def _initialize_detectors(self) -> None:
        """Initializes the OpenCV detectors from the process module."""
        try:
            self.face_detector, self.eye_detector, self.blob_detector = process.init_cv()
        except Exception as e:
            logging.error(f"Failed to initialize OpenCV detectors: {e}")
            # Optionally, disable UI elements that depend on this
            self.startButton.setEnabled(False)

    def _load_stylesheet(self) -> None:
        """Loads an external CSS stylesheet."""
        try:
            with open("style.css", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            logging.warning("style.css not found. Continuing without custom styles.")
        except Exception as e:
            logging.error(f"Error loading style.css: {e}")

    def _connect_signals(self) -> None:
        """Connects UI element signals (e.g., button clicks) to methods."""
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)

    def _open_capture_device(self) -> Optional[cv2.VideoCapture]:
        """Tries to open a video capture device, checking common APIs."""
        # Try DirectShow first (common on Windows for better performance)
        capture = cv2.VideoCapture(cv2.CAP_DSHOW)
        if capture.isOpened():
            logging.info("Opened camera using DirectShow (CAP_DSHOW).")
            return capture
        
        # Fallback to default API
        logging.warning("CAP_DSHOW failed. Trying default camera API (0).")
        capture = cv2.VideoCapture(0)
        if capture.isOpened():
            logging.info("Opened camera using default API (0).")
            return capture

        logging.error("Could not open any video stream.")
        return None

    def start_webcam(self) -> None:
        """Starts the webcam feed."""
        if self.camera_is_running:
            return

        self.capture = self._open_capture_device()
        if not self.capture:
            return  # Error was already logged in the helper

        self.camera_is_running = True
        self.timer.start(TIMER_INTERVAL)
        logging.info("Webcam started.")

    def stop_webcam(self) -> None:
        """Stops the webcam feed and clears displays."""
        if not self.camera_is_running:
            return

        self.camera_is_running = False
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None

        # Clear all image labels
        for label in self.image_labels.values():
            label.clear()
        logging.info("Webcam stopped.")

    def update_frame(self) -> None:
        """Called by the QTimer to process a new video frame."""
        if not self.camera_is_running or not self.capture:
            return

        ret, base_image = self.capture.read()
        if not ret:
            logging.warning("Failed to read frame from camera. Stopping webcam.")
            self.stop_webcam()
            return

        # Always display the base image
        self._display_image(base_image, MAIN_WINDOW_KEY)
        gray_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

        # --- Face Detection ---
        try:
            face_results = process.detect_face(base_image, gray_image, self.face_detector)
            face_frame, face_frame_gray, l_pos, r_pos, _, _ = face_results
        except Exception as e:
            logging.error(f"Error in face detection: {e}")
            self._clear_eye_labels()
            return

        if face_frame is None:
            self._clear_eye_labels()
            return  # No face detected

        # --- Eye Detection ---
        try:
            eye_results = process.detect_eyes(face_frame, face_frame_gray, l_pos, r_pos, self.eye_detector)
            left_eye_frame, right_eye_frame, left_eye_gray, right_eye_gray = eye_results
        except Exception as e:
            logging.error(f"Error in eye detection: {e}")
            self._clear_eye_labels()
            return
            
        # --- Process Each Eye ---
        self._process_eye(left_eye_frame, left_eye_gray, LEFT_EYE_KEY,
                          self.leftEyeCheckbox, self.leftEyeThreshold)
        self._process_eye(right_eye_frame, right_eye_gray, RIGHT_EYE_KEY,
                          self.rightEyeCheckbox, self.rightEyeThreshold)
        
        # --- (REMOVED) ---
        # Removed redundant display call that was here.
        # The main image is already displayed at the top of this method.

    def _process_eye(self,
                     eye_frame: Optional[NDArray],
                     eye_frame_gray: Optional[NDArray],
                     side: str,
                     checkbox: QCheckBox,
                     threshold_slider: QSlider) -> None:
        """Processes a single eye frame for blob detection and display."""
        if eye_frame is None or eye_frame_gray is None:
            self.image_labels[side].clear()
            return

        if checkbox.isChecked():
            threshold = threshold_slider.value()
            self._update_keypoints(eye_frame_gray, side, threshold)
            
            keypoints = self.tracking_state[side]['keypoints']
            if keypoints:
                try:
                    # process.draw_blobs is expected to modify eye_frame in-place
                    process.draw_blobs(eye_frame, keypoints)
                except Exception as e:
                    logging.error(f"Error drawing blobs for {side} eye: {e}")

        self._display_image(eye_frame, side)

    def _update_keypoints(self, frame_gray: NDArray, side: str, threshold: int) -> None:
        """Updates the tracking state (keypoints and area) for a given eye."""
        state = self.tracking_state[side]
        try:
            keypoints = process.process_eye(
                frame_gray,
                threshold,
                self.blob_detector,
                prevArea=state.get('area')  # Use .get for safety
            )
            
            if keypoints:
                state['keypoints'] = keypoints
                state['area'] = keypoints[0].size
            else:
                # Explicitly clear if no keypoints are found
                state['keypoints'] = None
                # Optionally reset area or keep last known
                # state['area'] = None 
        
        except Exception as e:
            logging.error(f"Error processing {side} eye: {e}")
            state['keypoints'] = None
            state['area'] = None

    def _display_image(self, img: Optional[NDArray], window_key: str) -> None:
        """
        Displays an OpenCV image (numpy array) in the specified QLabel.
        Handles Grayscale, BGR, and BGRA formats.
        """
        label = self.image_labels.get(window_key)
        if label is None:
            logging.warning(f"Invalid window key '{window_key}' for display.")
            return

        if img is None:
            label.clear()
            return

        try:
            # Ensure data is contiguous in memory for QImage
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)

            h, w, *channels = img.shape
            
            if img.ndim == 2:  # Grayscale
                bytes_per_line = w
                q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            
            elif img.ndim == 3:
                num_channels = channels[0]
                bytes_per_line = w * num_channels
                
                if num_channels == 3:  # BGR
                    q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                elif num_channels == 4:  # BGRA
                    q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_ARGB32)
                else:
                    logging.error(f"Unsupported 3D image with {num_channels} channels.")
                    label.clear()
                    return
            else:
                logging.error(f"Unsupported image dimensions: {img.ndim}")
                label.clear()
                return

            label.setPixmap(QPixmap.fromImage(q_img))
            label.setScaledContents(True)
        
        except Exception as e:
            logging.error(f"Error during image display for '{window_key}': {e}")
            label.clear()

    def _clear_eye_labels(self) -> None:
        """Clears the eye display QLabels."""
        self.image_labels[LEFT_EYE_KEY].clear()
        self.image_labels[RIGHT_EYE_KEY].clear()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handles the window close event."""
        logging.info("Closing application...")
        self.stop_webcam()
        event.accept()


def main() -> None:
    """Main entry point for the application."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    app = QApplication(sys.argv)
    window = EyeTrackerWindow()
    window.setWindowTitle("Eye Tracker")
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
