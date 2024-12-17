# Hand Gesture Controlled Fish Tank Lights 🚀

This project uses **MediaPipe** for hand gesture recognition and **Serial Communication** to control LED lights in a fish tank IoT circuit (simulated using Wokwi). 

The project recognizes predefined hand gestures using a trained deep learning model and sends commands via serial communication to control the LEDs.

---

## **Features**
- Recognizes hand gestures using **MediaPipe**.
- Sends signals to control LEDs based on gestures.
- Modular and clean code structure for flexibility.
- Works with **Wokwi Arduino Simulation** and real hardware setups.

---

## **File Structure**
```plaintext
light-controlling-hand-gesture/
│
├── data/                       # Collected landmarks data
│   ├── landmark_train.csv
│   ├── landmark_val.csv
│   ├── landmark_test.csv
│
├── models/
│   └── hand_gesture_model.h5   # Trained deep learning model
│
├── sign_images/                # Gesture reference images
│   ├── light1.jpg
│   ├── light2.jpg
│   ├── light3.jpg
│   ├── turn_off.jpg
│   └── turn_on.jpg
│
├── src/
│   ├── generate_landmark_data.py   # Collect landmarks for training
│   ├── train_model.py              # Train the hand gesture model
│   ├── test_model.py               # Test model predictions
│   ├── light_control.py            # Handles serial communication
│   ├── gesture_recognition.py      # Recognize gestures and control LEDs
│   ├── main.py                     # Entry point: runs the full system
│   └── test.py                     # Placeholder for general testing
│
├── .gitignore
├── README.md
└── requirements.txt
