import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import paho.mqtt.client as mqtt  # MQTT Library
import time

# MQTT Broker Configuration
MQTT_BROKER = "mqtt.eclipseprojects.io"
MQTT_PORT = 1883
TOPICS = {
    "LED": "fish_tank/device/led/command",
    "SERVO": "fish_tank/device/servo/command",
    "LEDBAR": "fish_tank/device/light/command",
}

# MQTT Client Initialization
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Load the hand gesture model
model = load_model("models/hand_gesture_model.h5")

# Class labels
LABELS = {
    0: "turn_off",
    1: "light1",
    2: "light2",
    3: "light3",
    4: "turn_on"
}   

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start capturing webcam feed
cap = cv2.VideoCapture(0)
print("Press 'q' to exit.")

last_sent = 0  # thời điểm gửi lần cuối, khởi tạo 0
SEND_INTERVAL = 2  # 5 giây mới gửi 1 lần

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []  
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Predict gesture
            prediction = model.predict(np.array([landmarks]))
            class_id = np.argmax(prediction)
            gesture = LABELS.get(class_id, "Unknown")

            # Display gesture on frame
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Chỉ gửi lệnh nếu đã quá 5 giây kể từ lần gửi trước
            current_time = time.time()
            if current_time - last_sent >= SEND_INTERVAL:
                if gesture == "turn_on":
                    mqtt_client.publish(TOPICS["LED"], "On")
                    print("Turn ON LED command sent via MQTT.")
                    last_sent = current_time
                elif gesture == "turn_off":
                    mqtt_client.publish(TOPICS["LED"], "Off")
                    print("Turn OFF LED command sent via MQTT.")
                    last_sent = current_time
                elif gesture == "light1":
                    mqtt_client.publish(TOPICS["LEDBAR"], "On")
                    print("Turn ON LED Bar command sent via MQTT.")
                    last_sent = current_time
                elif gesture == "light2":
                    mqtt_client.publish(TOPICS["LEDBAR"], "Off")
                    print("Turn OFF LED Bar command sent via MQTT.")
                    last_sent = current_time
                elif gesture == "light3":
                    mqtt_client.publish(TOPICS["SERVO"], "On")
                    print("Toggle Servo command sent via MQTT.")
                    last_sent = current_time

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
mqtt_client.disconnect()
