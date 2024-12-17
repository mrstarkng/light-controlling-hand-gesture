import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

# Load dữ liệu test
test_data = pd.read_csv("data/landmark_test.csv", header=None)
X_test = test_data.iloc[:, 1:].values  # Tọa độ landmark
y_test = test_data.iloc[:, 0].values  # Nhãn

# Load mô hình đã huấn luyện
model = load_model("models/hand_gesture_model.h5")

# Dự đoán nhãn
y_pred = np.argmax(model.predict(X_test), axis=1)

# Tính toán độ chính xác và hiển thị báo cáo phân loại
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác trên tập test: {accuracy * 100:.2f}%\n")
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred))
