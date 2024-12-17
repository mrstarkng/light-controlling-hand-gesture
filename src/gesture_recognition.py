import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from light_control import LightController  # Import LightController for serial communication

# Load hand gesture model
model = load_model("models/hand_gesture_model.h5")

# Class labels
LABELS = {
    0: "turn_off",
    1: "light1",
    2: "light2",
    3: "light3",
    4: "turn_on"
}

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize LightController for serial communication
light_controller = LightController(port="/dev/ttyUSB0", baudrate=9600)  # Replace with correct port

# Open webcam
cap = cv2.VideoCapture(0)
print("Nhấn 'q' để thoát.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand landmarks as input for the model
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Predict gesture using the loaded model
                prediction = model.predict(np.array([landmarks]))
                class_id = np.argmax(prediction)
                gesture = LABELS.get(class_id, "Unknown")

                # Display the recognized gesture on the screen
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Send serial commands based on the detected gesture
                if gesture == "turn_on":
                    light_controller.send_command("LIGHT_ON")
                elif gesture == "turn_off":
                    light_controller.send_command("LIGHT_OFF")
                elif gesture in ["light1", "light2", "light3"]:
                    light_controller.send_command(gesture.upper())  # LIGHT1, LIGHT2, LIGHT3

                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow("Gesture Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    light_controller.close()  # Close the serial connection
    cv2.destroyAllWindows()
