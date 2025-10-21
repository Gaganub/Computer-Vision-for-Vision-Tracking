import sys
import cv2
import logging
import numpy as np
from typing import Optional, Dict, Any
from numpy.typing import NDArray

import process  # uses OpenCV utilities

# --- PyQt5 Imports ---
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QCloseEvent
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QCheckBox, QSlider
from PyQt5.uic import loadUi

# --- Constants ---
FRAME_RATE = 30
FRAME_INTERVAL_MS = 1000 // FRAME_RATE
MAIN_VIEW = "main"
LEFT_VIEW = "left"
RIGHT_VIEW = "right"


class EyeTrackerApp(QMainWindow):
    """Primary application window handling camera feed and visual tracking."""

    def __init__(self) -> None:
        super().__init__()
        loadUi("GUImain.ui", self)
        self._apply_style()

        # Initialize camera components
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_active: bool = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)

        # Initialize OpenCV detectors
        self.face_cascade: Optional[cv2.CascadeClassifier] = None
        self.eye_cascade: Optional[cv2.CascadeClassifier] = None
        self.blob_detector: Optional[cv2.SimpleBlobDetector] = None
        self._setup_detectors()

        # Internal state
        self.eye_data: Dict[str, Dict[str, Any]] = {
            LEFT_VIEW: {"keypoints": None, "area": None},
            RIGHT_VIEW: {"keypoints": None, "area": None}
        }

        # Map QLabels for easy access
        self.display_boxes: Dict[str, QLabel] = {
            MAIN_VIEW: self.baseImage,
            LEFT_VIEW: self.leftEyeBox,
            RIGHT_VIEW: self.rightEyeBox
        }

        self._connect_ui_events()

    # ---------------------------------------------------------
    # Setup and Initialization
    # ---------------------------------------------------------
    def _setup_detectors(self) -> None:
        """Load OpenCV classifiers and detectors."""
        try:
            self.face_cascade, self.eye_cascade, self.blob_detector = process.init_cv()
        except Exception as exc:
            logging.error(f"OpenCV detector initialization failed: {exc}")
            self.startButton.setEnabled(False)

    def _apply_style(self) -> None:
        """Applies external stylesheet if available."""
        try:
            with open("style.css", "r") as css:
                self.setStyleSheet(css.read())
        except FileNotFoundError:
            logging.warning("style.css not found, proceeding with default look.")
        except Exception as e:
            logging.error(f"Error loading stylesheet: {e}")

    def _connect_ui_events(self) -> None:
        """Links UI button events."""
        self.startButton.clicked.connect(self._start_camera)
        self.stopButton.clicked.connect(self._stop_camera)

    # ---------------------------------------------------------
    # Camera and Timer Control
    # ---------------------------------------------------------
    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """Attempts to open the webcam device using various APIs."""
        cam = cv2.VideoCapture(cv2.CAP_DSHOW)
        if cam.isOpened():
            logging.info("Camera initialized via CAP_DSHOW.")
            return cam

        logging.warning("CAP_DSHOW failed, retrying with default API...")
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            logging.info("Camera opened with default backend.")
            return cam

        logging.error("Unable to access camera device.")
        return None

    def _start_camera(self) -> None:
        """Activates the webcam stream."""
        if self.is_active:
            return

        self.capture = self._open_camera()
        if not self.capture:
            return

        self.is_active = True
        self.timer.start(FRAME_INTERVAL_MS)
        logging.info("Camera stream started.")

    def _stop_camera(self) -> None:
        """Stops webcam stream and clears UI."""
        if not self.is_active:
            return

        self.is_active = False
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None

        for label in self.display_boxes.values():
            label.clear()

        logging.info("Camera stream stopped.")

    # ---------------------------------------------------------
    # Frame Update Logic
    # ---------------------------------------------------------
    def _update_frame(self) -> None:
        """Processes the next available frame from the webcam."""
        if not self.is_active or not self.capture:
            return

        ret, frame = self.capture.read()
        if not ret:
            logging.warning("Frame read failed; stopping stream.")
            self._stop_camera()
            return

        self._display_frame(frame, MAIN_VIEW)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            face_data = process.detect_face(frame, gray, self.face_cascade)
            face_frame, gray_face, left_region, right_region, *_ = face_data
        except Exception as e:
            logging.error(f"Face detection error: {e}")
            self._clear_eyes()
            return

        if face_frame is None:
            self._clear_eyes()
            return

        try:
            eyes = process.detect_eyes(face_frame, gray_face, left_region, right_region, self.eye_cascade)
            left, right, left_g, right_g = eyes
        except Exception as e:
            logging.error(f"Eye detection error: {e}")
            self._clear_eyes()
            return

        # Process each detected eye
        self._handle_eye(left, left_g, LEFT_VIEW, self.leftEyeCheckbox, self.leftEyeThreshold)
        self._handle_eye(right, right_g, RIGHT_VIEW, self.rightEyeCheckbox, self.rightEyeThreshold)

    # ---------------------------------------------------------
    # Eye Tracking Methods
    # ---------------------------------------------------------
    def _handle_eye(
        self,
        color_eye: Optional[NDArray],
        gray_eye: Optional[NDArray],
        label_key: str,
        checkbox: QCheckBox,
        slider: QSlider
    ) -> None:
        """Performs blob detection for a specific eye and updates display."""
        if color_eye is None or gray_eye is None:
            self.display_boxes[label_key].clear()
            return

        if checkbox.isChecked():
            threshold_value = slider.value()
            self._compute_keypoints(gray_eye, label_key, threshold_value)

            kps = self.eye_data[label_key]["keypoints"]
            if kps is not None:
                try:
                    process.draw_blobs(color_eye, kps)
                except Exception as e:
                    logging.error(f"Blob drawing failed for {label_key}: {e}")

        self._display_frame(color_eye, label_key)

    def _compute_keypoints(self, gray_eye: NDArray, key: str, threshold: int) -> None:
        """Updates blob keypoints for the given eye view."""
        state = self.eye_data[key]
        try:
            keypoints = process.process_eye(gray_eye, threshold, self.blob_detector, prevArea=state.get("area"))
            if keypoints is not None:
                state["keypoints"] = keypoints
                state["area"] = keypoints[0].size
            else:
                state["keypoints"] = None
        except Exception as e:
            logging.error(f"Keypoint processing failed for {key}: {e}")
            state["keypoints"], state["area"] = None, None

    # ---------------------------------------------------------
    # Display and Utility
    # ---------------------------------------------------------
    def _display_frame(self, frame: Optional[NDArray], target: str) -> None:
        """Renders a numpy frame into the corresponding QLabel."""
        box = self.display_boxes.get(target)
        if not box:
            logging.warning(f"Invalid display key: {target}")
            return

        if frame is None:
            box.clear()
            return

        try:
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)

            h, w = frame.shape[:2]

            if frame.ndim == 2:
                q_img = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
            elif frame.ndim == 3:
                ch = frame.shape[2]
                bytes_per_line = w * ch
                if ch == 3:
                    q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                elif ch == 4:
                    q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_ARGB32)
                else:
                    logging.error(f"Unsupported channel count: {ch}")
                    return
            else:
                logging.error(f"Unsupported image shape: {frame.shape}")
                return

            box.setPixmap(QPixmap.fromImage(q_img))
            box.setScaledContents(True)

        except Exception as e:
            logging.error(f"Display error ({target}): {e}")
            box.clear()

    def _clear_eyes(self) -> None:
        """Clears both left and right eye displays."""
        self.display_boxes[LEFT_VIEW].clear()
        self.display_boxes[RIGHT_VIEW].clear()

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:
        """Ensures camera release on window close."""
        logging.info("Application closing...")
        self._stop_camera()
        event.accept()


def main() -> None:
    """Program entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    app = QApplication(sys.argv)
    tracker = EyeTrackerApp()
    tracker.setWindowTitle("Eye Tracker System")
    tracker.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
