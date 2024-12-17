import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load mô hình
model = load_model("models/hand_gesture_model.h5")

# Class labels từ file YAML hoặc cố định
LABELS = {
    0: "turn_off",
    1: "light1",
    2: "light2",
    3: "light3",
    4: "turn_on"
}

# MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Mở webcam
cap = cv2.VideoCapture(0)
print("Nhấn 'q' để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển BGR sang RGB để MediaPipe xử lý
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Trích xuất tọa độ landmark
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Dự đoán nhãn cử chỉ
            prediction = model.predict(np.array([landmarks]))
            class_id = np.argmax(prediction)
            gesture = LABELS.get(class_id, "Unknown")

            # Hiển thị nhãn cử chỉ lên khung hình
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Vẽ landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hiển thị khung hình
    cv2.imshow("Gesture Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
