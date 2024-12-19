import cv2
import numpy as np
import paho.mqtt.client as mqtt
from tensorflow.keras.models import load_model
import mediapipe as mp

# MQTT Configuration
BROKER = "mqtt.eclipseprojects.io"
PORT = 1883
TOPIC_LED = "fish_tank/device/led/command"
TOPIC_SERVO = "fish_tank/device/servo/command"

# MQTT Client Initialization
client = mqtt.Client()
client.connect(BROKER, PORT, 60)

# Load Hand Gesture Model
model = load_model("models/hand_gesture_model.h5")

# Class labels  
LABELS = {
    0: "turn_off",
    1: "light1",
    2: "light2",
    3: "light3",
    4: "turn_on"
}

# MediaPipe Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start Video Capture
cap = cv2.VideoCapture(0)
print("Nhấn 'q' để thoát.")

def send_mqtt_command(gesture):
    """Send MQTT command based on the recognized gesture."""
    if gesture == "turn_off":
        client.publish(TOPIC_LED, "OFF")
    elif gesture == "turn_on":
        client.publish(TOPIC_LED, "ON")
    elif gesture == "light1":
        client.publish(TOPIC_SERVO, "ON")
    elif gesture == "light2":
        client.publish(TOPIC_SERVO, "OFF")
    print(f"Sent MQTT command: {gesture}")

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

            # Predict Gesture
            prediction = model.predict(np.array([landmarks]))
            class_id = np.argmax(prediction)
            gesture = LABELS.get(class_id, "Unknown")

            # Display Gesture
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            send_mqtt_command(gesture)  # Send MQTT Command

            # Draw Landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.disconnect()
