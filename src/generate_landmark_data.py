import os
import cv2
import csv
import yaml
import numpy as np
import mediapipe as mp

# Class ghi dữ liệu landmark vào file CSV
class HandDatasetWriter:
    def __init__(self, filepath):
        self.csv_file = open(filepath, "a")  # Mở file CSV ở chế độ append
        self.file_writer = csv.writer(self.csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    def add(self, hand, label):
        # Thêm một dòng dữ liệu mới vào file CSV
        self.file_writer.writerow([label, *np.array(hand).flatten().tolist()])
    
    def close(self):
        self.csv_file.close()

# Class phát hiện landmarks từ MediaPipe
class HandLandmarksDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(False, max_num_hands=1, min_detection_confidence=0.5)
    
    def detectHand(self, frame):
        hands = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB
        results = self.detector.process(rgb_frame)  # Nhận diện bàn tay bằng MediaPipe

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                for landmark in hand_landmarks.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    hand.extend([x, y, z])
                hands.append(hand)  # Thêm tọa độ landmarks của bàn tay
        return hands, results.multi_hand_landmarks

# Hàm đọc file cấu hình YAML và trả về từ điển class
def label_dict_from_config_file(relative_path):
    with open(relative_path, "r") as f:
        label_tag = yaml.full_load(f)["gestures"]
    return label_tag

# Hàm chính để thu thập dữ liệu landmark
def run(data_path, sign_img_path, split="train", resolution=(1280, 720)):
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(sign_img_path, exist_ok=True)
    
    dataset_path = f"./{data_path}/landmark_{split}.csv"
    hand_dataset = HandDatasetWriter(dataset_path)
    hand_detector = HandLandmarksDetector()
    
    cam = cv2.VideoCapture(0)
    cam.set(3, resolution[0])  # Đặt chiều rộng khung hình
    cam.set(4, resolution[1])  # Đặt chiều cao khung hình
    
    current_letter = None
    saved_frame = None
    cannot_switch_char = False
    status_text = "Press a character to record"

    LABEL_TAG = label_dict_from_config_file("data/hand.yaml")

    print("Nhấn các phím tương ứng với cử chỉ cần ghi: (a, b, c...)")
    print("Nhấn 'q' để thoát chương trình.")

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        
        hands, landmarks = hand_detector.detectHand(frame)
        annotated_image = frame.copy()

        # Hiển thị thông báo trạng thái
        if current_letter is None:
            status_text = "Press a character to start recording"

        if current_letter:
            label = ord(current_letter) - ord("a")
            status_text = f"Recording {LABEL_TAG[label]}, press {current_letter} again to stop"

            if hands:  # Ghi dữ liệu khi phát hiện bàn tay
                hand_dataset.add(hands[0], label)
                saved_frame = frame.copy()
        
        # Nhấn phím để thay đổi cử chỉ hoặc thoát chương trình
        key = cv2.waitKey(1)
        if key != -1:
            char = chr(key)
            if char == "q":  # Thoát chương trình
                break
            elif current_letter is None and char.isalpha():  # Bắt đầu ghi dữ liệu
                current_letter = char
            elif current_letter == char:  # Dừng ghi dữ liệu
                if saved_frame is not None:
                    cv2.imwrite(f"{sign_img_path}/{LABEL_TAG[label]}.jpg", saved_frame)
                current_letter = None
                saved_frame = None

        # Hiển thị thông tin lên khung hình
        cv2.putText(annotated_image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collect Hand Landmarks", annotated_image)

    cam.release()
    cv2.destroyAllWindows()
    hand_dataset.close()

if __name__ == "__main__":
    data_path = "./data"
    sign_img_path = "./sign_images"
    run(data_path, sign_img_path, split="train")
