"""
Hand Gesture Media Control - Simplified Version
All-in-one application for controlling media using hand gestures.
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import keyboard
import time
from typing import List, Tuple, Optional, Any
from collections import Counter, deque
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER


# ==================== CONFIGURATION ====================
# Hand Detection
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.7

# Gesture Thresholds
PINCH_THRESHOLD = 0.06
VOLUME_MIN_DISTANCE = 0.03
VOLUME_MAX_DISTANCE = 0.18

# UI Settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# ==================== HAND DETECTOR ====================
class HandDetector:
    """Detects hand landmarks using MediaPipe."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands  # type: ignore
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.mp_drawing = mp.solutions.drawing_utils  # type: ignore

    def detect(self, frame: np.ndarray) -> Tuple[Any, List[List[Tuple[float, float, float]]]]:
        """Detect hands and return landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                landmarks_list.append(landmarks)
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        
        return frame, landmarks_list

    def release(self):
        self.hands.close()


# ==================== GESTURE RECOGNIZER ====================
class GestureRecognizer:
    """Recognizes hand gestures from landmarks."""

    def __init__(self):
        self.gesture_history = []
        self.history_size = 7

    @staticmethod
    def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate distance between two points."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def is_pinch(self, lm: List[Tuple[float, float, float]]) -> bool:
        """Check if thumb and index are pinched."""
        return self.distance((lm[4][0], lm[4][1]), (lm[8][0], lm[8][1])) < PINCH_THRESHOLD

    def is_thumb_up(self, lm: List[Tuple[float, float, float]]) -> bool:
        """Check thumbs up gesture."""
        return (lm[4][1] < lm[8][1] - 0.08 and lm[4][1] < lm[12][1] - 0.08 and
                lm[8][1] > lm[6][1] and lm[12][1] > lm[10][1])

    def is_thumb_down(self, lm: List[Tuple[float, float, float]]) -> bool:
        """Check thumbs down gesture."""
        return (lm[4][1] > lm[8][1] + 0.08 and lm[4][1] > lm[12][1] + 0.08 and
                lm[8][1] > lm[6][1] and lm[12][1] > lm[10][1])

    def is_peace_sign(self, lm: List[Tuple[float, float, float]]) -> bool:
        """Check peace sign (V) gesture."""
        index_ext = lm[8][1] < lm[6][1]
        middle_ext = lm[12][1] < lm[10][1]
        ring_fold = lm[16][1] > lm[14][1]
        pinky_fold = lm[20][1] > lm[18][1]
        separated = self.distance((lm[8][0], lm[8][1]), (lm[12][0], lm[12][1])) > 0.04
        return index_ext and middle_ext and ring_fold and pinky_fold and separated

    def is_pointing(self, lm: List[Tuple[float, float, float]]) -> bool:
        """Check pointing gesture."""
        return (lm[8][1] < lm[6][1] and lm[12][1] > lm[10][1] and
                lm[16][1] > lm[14][1] and lm[20][1] > lm[18][1])

    def is_fist(self, lm: List[Tuple[float, float, float]]) -> bool:
        """Check fist gesture."""
        return all([lm[8][1] > lm[6][1], lm[12][1] > lm[10][1],
                    lm[16][1] > lm[14][1], lm[20][1] > lm[18][1]])

    def is_palm_open(self, lm: List[Tuple[float, float, float]]) -> bool:
        """Check open palm gesture."""
        palm = (lm[0][0], lm[0][1])
        distances = [self.distance((lm[i][0], lm[i][1]), palm) for i in [4, 8, 12, 16, 20]]
        return all(d > 0.15 for d in distances)

    def recognize(self, landmarks: List[Tuple[float, float, float]]) -> str:
        """Recognize gesture from landmarks."""
        if self.is_pinch(landmarks):
            gesture = "pinch"
        elif self.is_thumb_up(landmarks):
            gesture = "thumbs_up"
        elif self.is_thumb_down(landmarks):
            gesture = "thumbs_down"
        elif self.is_peace_sign(landmarks):
            gesture = "peace"
        elif self.is_pointing(landmarks):
            gesture = "point"
        elif self.is_fist(landmarks):
            gesture = "fist"
        elif self.is_palm_open(landmarks):
            gesture = "palm"
        else:
            gesture = "none"

        # Apply smoothing
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)

        if len(self.gesture_history) >= 3:
            recent = self.gesture_history[-3:]
            if len(set(recent)) == 1:
                return recent[0]
            counts = Counter(self.gesture_history)
            most_common = counts.most_common(1)[0]
            if most_common[1] >= len(self.gesture_history) * 0.5:
                return most_common[0]

        return "none"

    def get_volume_level(self, landmarks: List[Tuple[float, float, float]]) -> float:
        """Get volume level from pinch distance."""
        distance = self.distance((landmarks[4][0], landmarks[4][1]), (landmarks[8][0], landmarks[8][1]))
        volume = (distance - VOLUME_MIN_DISTANCE) / (VOLUME_MAX_DISTANCE - VOLUME_MIN_DISTANCE)
        return max(0.0, min(1.0, volume))


# ==================== MEDIA CONTROLLER ====================
class MediaController:
    """Controls system media and volume."""

    def __init__(self):
        self.last_action_time = {}
        self._init_audio()

    def _init_audio(self):
        """Initialize audio volume control."""
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, 0, None)  # type: ignore
            self.volume_control = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as e:
            print(f"Warning: Could not initialize volume control: {e}")
            self.volume_control = None

    def can_execute(self, gesture: str) -> bool:
        """Check if gesture can be executed (debouncing)."""
        current_time = time.time()
        if gesture not in self.last_action_time:
            self.last_action_time[gesture] = current_time
            return True
        if current_time - self.last_action_time[gesture] >= 1.0:
            self.last_action_time[gesture] = current_time
            return True
        return False

    def set_volume(self, level: float):
        """Set system volume."""
        if self.volume_control:
            try:
                self.volume_control.SetMasterVolumeLevelScalar(max(0.0, min(1.0, level)), None)  # type: ignore
            except Exception as e:
                print(f"Error setting volume: {e}")

    def volume_up(self):
        """Increase volume."""
        if self.volume_control:
            try:
                current = self.volume_control.GetMasterVolumeLevelScalar()  # type: ignore
                self.volume_control.SetMasterVolumeLevelScalar(min(1.0, current + 0.05), None)  # type: ignore
                print("Volume up")
            except Exception as e:
                print(f"Error: {e}")

    def volume_down(self):
        """Decrease volume."""
        if self.volume_control:
            try:
                current = self.volume_control.GetMasterVolumeLevelScalar()  # type: ignore
                self.volume_control.SetMasterVolumeLevelScalar(max(0.0, current - 0.05), None)  # type: ignore
                print("Volume down")
            except Exception as e:
                print(f"Error: {e}")

    def play_pause(self):
        """Play/Pause media."""
        try:
            keyboard.press_and_release("space")
            print("Play/Pause")
        except Exception as e:
            print(f"Error: {e}")

    def next_track(self):
        """Next track."""
        try:
            keyboard.press_and_release("alt+right")
            print("Next track")
        except Exception as e:
            print(f"Error: {e}")

    def previous_track(self):
        """Previous track."""
        try:
            keyboard.press_and_release("alt+left")
            print("Previous track")
        except Exception as e:
            print(f"Error: {e}")

    def mute(self):
        """Toggle mute."""
        if self.volume_control:
            try:
                current = self.volume_control.GetMasterVolumeLevelScalar()  # type: ignore
                new_volume = 0.0 if current > 0 else 0.5
                self.volume_control.SetMasterVolumeLevelScalar(new_volume, None)  # type: ignore
                print(f"Mute toggled: {new_volume}")
            except Exception as e:
                print(f"Error: {e}")

    def get_volume(self) -> Optional[float]:
        """Get current volume."""
        if self.volume_control:
            try:
                return self.volume_control.GetMasterVolumeLevelScalar()  # type: ignore
            except:
                return None
        return None

    def handle_gesture(self, gesture: str):
        """Handle gesture with debouncing."""
        if not self.can_execute(gesture):
            return

        actions = {
            "thumbs_up": self.volume_up,
            "thumbs_down": self.volume_down,
            "peace": self.play_pause,
            "point": self.next_track,
            "palm": self.previous_track,
            "fist": self.mute,
        }

        if gesture in actions:
            actions[gesture]()


# ==================== STREAMLIT UI ====================
def initialize_session_state():
    """Initialize session state."""
    if "detector" not in st.session_state:
        st.session_state.detector = HandDetector()
    if "recognizer" not in st.session_state:
        st.session_state.recognizer = GestureRecognizer()
    if "controller" not in st.session_state:
        st.session_state.controller = MediaController()


def add_gesture_overlay(frame: np.ndarray, gesture: str, confidence: float) -> np.ndarray:
    """Add gesture info overlay."""
    cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (400, 80), (0, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture.upper()}", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
    return frame


def main():
    """Main application."""
    st.set_page_config(page_title="Hand Gesture Media Control", page_icon="🖐️", layout="wide")
    
    st.title("🖐️ Hand Gesture Media Control")
    st.markdown("Control your media with hand gestures!")

    initialize_session_state()

    # Sidebar
    st.sidebar.header("⚙️ Controls")
    st.sidebar.info("""
    **Gestures:**
    - 👆 Thumbs Up → Volume Up
    - 👎 Thumbs Down → Volume Down
    - ✌️ Peace Sign → Play/Pause
    - ☝️ Pointing → Next Track
    - ✋ Open Palm → Previous Track
    - ✊ Fist → Mute
    - 🤏 Pinch → Volume Control
    """)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📹 Live Feed")
        frame_placeholder = st.empty()

    with col2:
        st.subheader("📊 Status")
        gesture_text = st.empty()
        confidence_text = st.empty()
        volume_text = st.empty()

    # Camera control
    run = st.button("▶️ Start Camera")
    stop = st.button("⏹️ Stop")

    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        detector = st.session_state.detector
        recognizer = st.session_state.recognizer
        controller = st.session_state.controller

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)
            
            # Detect and recognize
            frame, landmarks_list = detector.detect(frame)
            
            gesture = "none"
            confidence = 0.0

            if landmarks_list:
                landmarks = landmarks_list[0]
                gesture = recognizer.recognize(landmarks)

                # Handle gestures
                if gesture != "pinch" and gesture != "none":
                    controller.handle_gesture(gesture)

                if gesture == "pinch":
                    volume = recognizer.get_volume_level(landmarks)
                    controller.set_volume(volume)
                    confidence = volume
                else:
                    confidence = 1.0 if gesture != "none" else 0.0

            # Add overlay
            frame = add_gesture_overlay(frame, gesture, confidence)

            # Display
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            with col2:
                gesture_text.metric("Gesture", gesture.upper())
                confidence_text.metric("Confidence", f"{confidence:.2f}")
                vol = controller.get_volume()
                if vol:
                    volume_text.metric("Volume", f"{vol * 100:.0f}%")

            if stop:
                break

        cap.release()


if __name__ == "__main__":
    main()
