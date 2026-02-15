import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
from playsound import playsound
import time

# --- Settings ---
EAR_THRESHOLD = 0.25    # Eye aspect ratio threshold
CONSEC_FRAMES = 20      # Frames eyes must be closed to trigger alarm
counter = 0
ALARM_ON = False
alarm_thread = None
ALARM_FILE = "alarm.wav"  # Short WAV file (~0.5-1s)

# --- Load MediaPipe FaceLandmarker Model ---
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- Eye landmark indices ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# --- Alarm function ---
def sound_alarm():
    """Plays alarm repeatedly while ALARM_ON is True"""
    global ALARM_ON
    while ALARM_ON:
        try:
            playsound(ALARM_FILE)  # short WAV recommended
        except Exception as e:
            print(f"Error playing alarm: {e}")
        time.sleep(0.1)  # small pause

# --- Calculate EAR ---
def calculate_EAR(eye_points, landmarks, w, h, frame):
    coords = []
    for point in eye_points:
        x = int(landmarks[point].x * w)
        y = int(landmarks[point].y * h)
        coords.append((x, y))
    coords = np.array(coords)

    # Draw eye contour
    cv2.polylines(frame, [coords], isClosed=True, color=(0, 255, 0), thickness=1)

    # EAR formula
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    C = np.linalg.norm(coords[0] - coords[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- Initialize camera ---
cap = cv2.VideoCapture(1)  # Use default webcam

print("Driver Monitoring System Started. Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]

        # Calculate EAR for both eyes
        left_ear = calculate_EAR(LEFT_EYE, landmarks, w, h, frame)
        right_ear = calculate_EAR(RIGHT_EYE, landmarks, w, h, frame)
        ear = (left_ear + right_ear) / 2.0

        # --- Drowsiness check ---
        if ear < EAR_THRESHOLD:
            counter += 1
            if counter >= CONSEC_FRAMES:
                # Flash red overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                # Display alert text
                cv2.putText(frame, "!!! EMERGENCY ALERT !!!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                # Start alarm if not already running
                if not ALARM_ON:
                    ALARM_ON = True
                    if alarm_thread is None or not alarm_thread.is_alive():
                        alarm_thread = threading.Thread(target=sound_alarm, daemon=True)
                        alarm_thread.start()
        else:
            # Eyes open → reset counter and stop alarm
            counter = 0
            ALARM_ON = False

        # Display EAR on screen
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video frame
    cv2.imshow("Driver Monitoring System", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
ALARM_ON = False
