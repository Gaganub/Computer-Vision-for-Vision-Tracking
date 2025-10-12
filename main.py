import sys
import cv2
import numpy as np
import process
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi

FPS = 30
TIMER_INTERVAL = 1000 // FPS

class EyeTrackerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('GUImain.ui', self)
        self._load_stylesheet()

        self.capture = None
        self.camera_is_running = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.face_detector, self.eye_detector, self.blob_detector = process.init_cv()

        self.tracking_state = {
            'left': {'keypoints': None, 'area': None},
            'right': {'keypoints': None, 'area': None}
        }

        self.image_labels = {
            'main': self.baseImage,
            'left': self.leftEyeBox,
            'right': self.rightEyeBox
        }

        self._connect_signals()

    def _load_stylesheet(self):
        try:
            with open("style.css", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print("Warning: style.css not found.")

    def _connect_signals(self):
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)

    def start_webcam(self):
        if self.camera_is_running:
            return

        self.capture = cv2.VideoCapture(cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            print("Error: Could not open video stream.")
            return

        self.camera_is_running = True
        self.timer.start(TIMER_INTERVAL)

    def stop_webcam(self):
        if not self.camera_is_running:
            return

        self.camera_is_running = False
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None

        for label in self.image_labels.values():
            label.clear()

    def update_frame(self):
        ret, base_image = self.capture.read()
        if not ret:
            self.stop_webcam()
            return

        self._display_image(base_image, 'main')
        gray_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

        face_results = process.detect_face(base_image, gray_image, self.face_detector)
        face_frame, face_frame_gray, l_pos, r_pos, _, _ = face_results

        if face_frame is None:
            self._clear_eye_labels()
            return

        eye_results = process.detect_eyes(face_frame, face_frame_gray, l_pos, r_pos, self.eye_detector)
        left_eye_frame, right_eye_frame, left_eye_gray, right_eye_gray = eye_results

        self._process_eye(left_eye_frame, left_eye_gray, 'left', self.leftEyeCheckbox, self.leftEyeThreshold)
        self._process_eye(right_eye_frame, right_eye_gray, 'right', self.rightEyeCheckbox, self.rightEyeThreshold)

        if self.pupilsCheckbox.isChecked():
            self._display_image(base_image, 'main')

    def _process_eye(self, eye_frame, eye_frame_gray, side, checkbox, threshold_slider):
        if eye_frame is None:
            self.image_labels[side].clear()
            return

        if checkbox.isChecked():
            threshold = threshold_slider.value()
            self._update_keypoints(eye_frame_gray, side, threshold)
            keypoints = self.tracking_state[side]['keypoints']
            if keypoints:
                process.draw_blobs(eye_frame, keypoints)

        self._display_image(eye_frame, side)

    def _update_keypoints(self, frame_gray, side, threshold):
        state = self.tracking_state[side]
        keypoints = process.process_eye(
            frame_gray,
            threshold,
            self.blob_detector,
            prevArea=state['area']
        )
        if keypoints:
            state['keypoints'] = keypoints
            state['area'] = keypoints[0].size

    def _display_image(self, img, window_key):
        label = self.image_labels.get(window_key)
        if img is None or label is None:
            if label:
                label.clear()
            return

        img = np.ascontiguousarray(img)
        h, w, *ch = img.shape
        
        if img.ndim == 2:
            q_img = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        elif ch[0] == 3:
            q_img = QImage(img.data, w, h, w * 3, QImage.Format_RGB888).rgbSwapped()
        elif ch[0] == 4:
            q_img = QImage(img.data, w, h, w * 4, QImage.Format_ARGB32)
        else:
            return

        label.setPixmap(QPixmap.fromImage(q_img))
        label.setScaledContents(True)

    def _clear_eye_labels(self):
        self.image_labels['left'].clear()
        self.image_labels['right'].clear()

    def closeEvent(self, event):
        self.stop_webcam()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = EyeTrackerWindow()
    window.setWindowTitle("Eye Tracker")
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
