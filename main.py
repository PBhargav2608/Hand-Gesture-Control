# Hand Gesture Control
import threading
import traceback
import time
import sys
from math import hypot
from ctypes import cast, POINTER

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import screen_brightness_control as sbc
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from PyQt5 import QtWidgets, QtGui, QtCore


class ControlWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Advanced Hand Gesture Control')
        self.setGeometry(300, 300, 500, 400)

        mainLayout = QtWidgets.QVBoxLayout()

        controlBox = QtWidgets.QGroupBox("Control Options")
        controlLayout = QtWidgets.QVBoxLayout()
        self.volumeCheckbox = QtWidgets.QCheckBox('Control Volume (Right Hand)')
        self.volumeCheckbox.setChecked(True)
        self.brightnessCheckbox = QtWidgets.QCheckBox('Control Brightness (Left Hand)')
        self.brightnessCheckbox.setChecked(True)
        self.mouseCheckbox = QtWidgets.QCheckBox('Control Mouse Pointer')
        controlLayout.addWidget(self.volumeCheckbox)
        controlLayout.addWidget(self.brightnessCheckbox)
        controlLayout.addWidget(self.mouseCheckbox)
        controlBox.setLayout(controlLayout)

        displayBox = QtWidgets.QGroupBox("Display Options")
        displayLayout = QtWidgets.QVBoxLayout()
        self.toggleGraphCheckbox = QtWidgets.QCheckBox('Show Hand Landmarks')
        self.toggleGraphCheckbox.setChecked(True)
        self.maximizeButton = QtWidgets.QPushButton('Maximize Video')
        self.maximizeButton.clicked.connect(self.toggleMaximize)
        self.isMaximized = False
        displayLayout.addWidget(self.toggleGraphCheckbox)
        displayLayout.addWidget(self.maximizeButton)
        displayBox.setLayout(displayLayout)

        self.fixButton = QtWidgets.QPushButton('Lock Control')
        self.fixButton.setCheckable(True)
        self.fixButton.clicked.connect(self.toggleFixButton)

        self.statusLabel = QtWidgets.QLabel("Status: Volume: 0% Brightness: 0%")
        self.statusLabel.setAlignment(QtCore.Qt.AlignCenter)

        mainLayout.addWidget(controlBox)
        mainLayout.addWidget(displayBox)
        mainLayout.addWidget(self.fixButton)
        mainLayout.addWidget(self.statusLabel)

        self.setLayout(mainLayout)

        # optional: set icon if you have it
        try:
            self.setWindowIcon(QtGui.QIcon('hand_icon.png'))
        except Exception:
            pass

        self.setStyleSheet("""
            QWidget {
                font-size: 14px;
                font-family: Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid gray;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                font-size: 14px;
                padding: 8px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
            QPushButton:checked {
                background-color: #f44336;
            }
            QPushButton:checked:hover {
                background-color: #d32f2f;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #999;
                background: white;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #4CAF50;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #555;
            }
            QLabel {
                font-size: 16px;
                color: #333;
            }
        """)
        self.show()

    def toggleMaximize(self):
        self.isMaximized = not self.isMaximized
        self.maximizeButton.setText('Minimize Video' if self.isMaximized else 'Maximize Video')

    def toggleFixButton(self):
        if self.fixButton.isChecked():
            self.fixButton.setText('Unlock Control')
        else:
            self.fixButton.setText('Lock Control')

    def updateStatus(self, volume, brightness):
        self.statusLabel.setText(f"Status: Volume: {volume}% Brightness: {brightness}%")


class HandControl:
    def __init__(self, gui):
        self.gui = gui
        self.cap = cv2.VideoCapture(0)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mpDraw = mp.solutions.drawing_utils

        # audio setup (Windows + pycaw)
        try:
            devices = AudioUtilities.GetSpeakers()
            # IMPORTANT: use iid attribute
            interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            try:
                self.volMin, self.volMax = self.volume.GetVolumeRange()[:2]
            except Exception:
                self.volMin, self.volMax = -65.25, 0.0
            # print debug info
            try:
                print("pycaw initialized. Current scalar:", self.volume.GetMasterVolumeLevelScalar())
            except Exception:
                print("pycaw initialized but couldn't read scalar, range:", self.volMin, self.volMax)
        except Exception as e:
            print("pycaw init failed:", repr(e))
            self.volume = None
            self.volMin, self.volMax = -65.25, 0.0  # fallback typical range

        self.brightnessMin, self.brightnessMax = 0, 100
        self.screenWidth, self.screenHeight = pyautogui.size()

        # gesture click state
        self.last_left_click = 0.0
        self.last_right_click = 0.0
        self.last_double_click = 0.0
        self.left_click_cooldown = 0.4   # seconds between clicks
        self.double_click_window = 0.4   # time window to treat two clicks as double
        self.dragging = False
        self.drag_start_time = None
        self.hold_threshold = 0.45       # seconds to treat pinch as hold (drag)
        self.pinch_threshold = 0.04      # normalized distance threshold for pinch (tweak)

        # volume set throttling
        self.last_set_volume_scalar = None
        self.volume_set_cooldown = 0.08  # seconds
        self._last_volume_set_time = 0.0

        # initialize safely
        try:
            self.currentVolume = int(np.interp(self.get_current_volume(), [self.volMin, self.volMax], [0, 100]))
        except Exception:
            self.currentVolume = 0
        try:
            self.currentBrightness = sbc.get_brightness()[0]
        except Exception:
            self.currentBrightness = 50

    def get_current_volume(self):
        if self.volume is None:
            return self.volMin
        # try scalar getter when possible; fallback to dB
        try:
            return self.volume.GetMasterVolumeLevel()
        except Exception:
            try:
                scalar = self.volume.GetMasterVolumeLevelScalar()
                return np.interp(scalar, [0.0, 1.0], [self.volMin, self.volMax])
            except Exception:
                return self.volMin

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                print("[handcontrol] camera read failed or camera closed")
                break

            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            leftHand, rightHand = None, None
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    if label == "Left":
                        leftHand = hand_landmarks
                    else:
                        rightHand = hand_landmarks

                    if self.gui.toggleGraphCheckbox.isChecked():
                        self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

            if not self.gui.fixButton.isChecked():
                if leftHand and self.gui.brightnessCheckbox.isChecked():
                    self.currentBrightness = self.control_brightness(img, leftHand)
                if rightHand and self.gui.volumeCheckbox.isChecked():
                    self.currentVolume = self.control_volume(img, rightHand)
                if (leftHand or rightHand) and self.gui.mouseCheckbox.isChecked():
                    self.control_mouse(img, leftHand or rightHand)

            # update GUI label
            try:
                self.gui.updateStatus(self.currentVolume, self.currentBrightness)
            except Exception:
                pass

            # window sizing
            if self.gui.isMaximized:
                cv2.namedWindow('Hand Control', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Hand Control', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow('Hand Control', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Hand Control', 1280, 720)

            cv2.imshow('Hand Control', img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('f'):
                self.gui.toggleMaximize()

        self.cap.release()
        cv2.destroyAllWindows()

    def control_brightness(self, img, hand_landmarks):
        x1, y1 = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
        x2, y2 = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
        length = hypot(x2 - x1, y2 - y1)
        brightness = np.interp(length, [0.02, 0.20], [self.brightnessMin, self.brightnessMax])
        brightness = int(np.clip(brightness, self.brightnessMin, self.brightnessMax))
        try:
            sbc.set_brightness(brightness)
        except Exception:
            pass
        cv2.putText(img, f"Brightness: {brightness}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return brightness

    def control_volume(self, img, hand_landmarks):
        # thumb tip and index tip already used elsewhere; we use the same indices
        x1, y1 = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
        x2, y2 = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
        length = hypot(x2 - x1, y2 - y1)

        # map length -> scalar 0.0..1.0 (tweak the input range to your setup)
        vol_scalar = np.interp(length, [0.02, 0.20], [0.0, 1.0])
        vol_scalar = float(np.clip(vol_scalar, 0.0, 1.0))

        now = time.time()
        # throttle real system calls to avoid spamming
        if self.volume is not None and (self.last_set_volume_scalar is None or abs(self.last_set_volume_scalar - vol_scalar) > 0.01) and (now - self._last_volume_set_time) > self.volume_set_cooldown:
            try:
                # prefer scalar API (most reliable)
                self.volume.SetMasterVolumeLevelScalar(vol_scalar, None)
                self.last_set_volume_scalar = vol_scalar
                self._last_volume_set_time = now
                # debug occasional print (every ~5s) so console isn't flooded
                if int(now) % 5 == 0:
                    print(f"[handcontrol] SetMasterVolumeLevelScalar -> {vol_scalar:.2f}")
            except Exception:
                # fallback to dB-level set
                try:
                    vol_db = np.interp(length, [0.02, 0.20], [self.volMin, self.volMax])
                    self.volume.SetMasterVolumeLevel(vol_db, None)
                    self.last_set_volume_scalar = vol_scalar
                    self._last_volume_set_time = now
                except Exception:
                    pass

        # display percentage
        volPer = int(vol_scalar * 100)
        cv2.putText(img, f"Volume: {volPer}%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return volPer

    def control_mouse(self, img, hand_landmarks):
        # landmark positions (normalized 0..1)
        tx, ty = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y  # thumb tip
        ix, iy = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y  # index tip
        mx, my = hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y # middle tip

        # screen coords (index finger used for pointer)
        screenX = int(np.interp(ix, [0, 1], [0, self.screenWidth]))
        screenY = int(np.interp(iy, [0, 1], [0, self.screenHeight]))

        # move cursor (smoothing optional)
        try:
            pyautogui.moveTo(screenX, screenY)
        except Exception:
            pass
        cv2.putText(img, f"Mouse: ({screenX}, {screenY})", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # compute distances (normalized)
        def dist(a, b, c, d):
            return ((a - c) ** 2 + (b - d) ** 2) ** 0.5

        thumb_index_dist = dist(tx, ty, ix, iy)
        index_middle_dist = dist(ix, iy, mx, my)

        now = time.time()

        # ----- LEFT CLICK (thumb+index pinch) -----
        if thumb_index_dist < self.pinch_threshold:
            # start dragging if held longer than hold_threshold
            if not self.dragging:
                # pinch started
                if self.drag_start_time is None:
                    self.drag_start_time = now
                held = (now - self.drag_start_time) >= self.hold_threshold

                if held:
                    # start drag
                    try:
                        pyautogui.mouseDown(button='left')
                    except Exception:
                        pass
                    self.dragging = True
                    cv2.putText(img, "Drag start", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # quick pinch -> treat as click if cooldown passed
                    if (now - self.last_left_click) > self.left_click_cooldown:
                        # check for double-click
                        if (now - self.last_left_click) < self.double_click_window:
                            try:
                                pyautogui.doubleClick()
                            except Exception:
                                pass
                            self.last_double_click = now
                        else:
                            try:
                                pyautogui.click()
                            except Exception:
                                pass
                            # store last click time
                            self.last_left_click = now
                        # small delay to avoid multiple triggers in same frame
                        time.sleep(0.05)
            else:
                # currently dragging -> keep dragging (cursor already moves)
                cv2.putText(img, "Dragging...", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # pinch released
            self.drag_start_time = None
            if self.dragging:
                # finish drag
                try:
                    pyautogui.mouseUp(button='left')
                except Exception:
                    pass
                self.dragging = False
                cv2.putText(img, "Drag end", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ----- RIGHT CLICK (index+middle pinch) -----
        if index_middle_dist < self.pinch_threshold:
            if (now - self.last_right_click) > self.left_click_cooldown:
                try:
                    pyautogui.click(button='right')
                except Exception:
                    pass
                self.last_right_click = now
                time.sleep(0.05)  # tiny debounce to avoid repeats

        # optional: visual debug of distances
        cv2.putText(img, f"t-i: {thumb_index_dist:.3f} i-m: {index_middle_dist:.3f}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def thread_wrapper(fn):
    def run_and_log():
        try:
            print("[handcontrol] thread started")
            fn()
            print("[handcontrol] thread finished normally")
        except Exception:
            print("[handcontrol] exception in thread:", file=sys.stderr)
            traceback.print_exc()
            with open("handcontrol_error.log", "a") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                traceback.print_exc(file=f)
    return run_and_log


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = ControlWindow()
    handControl = HandControl(gui)

    t = threading.Thread(target=thread_wrapper(handControl.run), daemon=True)
    t.start()
    print("[main] GUI starting (app.exec_())")
    sys.exit(app.exec_())