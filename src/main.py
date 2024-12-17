from gesture_recognition import recognize_gesture
from light_control import control_light

def main():
    print("Hệ thống điều khiển đèn bằng cử chỉ tay")
    while True:
        gesture = recognize_gesture()
        if gesture == "turn_on":
            control_light("ON")
            print("Đèn đã bật")
        elif gesture == "turn_off":
            control_light("OFF")
            print("Đèn đã tắt")

if __name__ == "__main__":
    main()
