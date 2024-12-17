import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load dữ liệu
def load_data(train_path, val_path):
    train_data = pd.read_csv(train_path, header=None)
    val_data = pd.read_csv(val_path, header=None)

    X_train = train_data.iloc[:, 1:].values  # Tọa độ landmarks
    y_train = train_data.iloc[:, 0].values  # Nhãn cử chỉ

    X_val = val_data.iloc[:, 1:].values
    y_val = val_data.iloc[:, 0].values

    return X_train, to_categorical(y_train), X_val, to_categorical(y_val)

# Xây dựng mô hình MLP
def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Huấn luyện mô hình
X_train, y_train, X_val, y_val = load_data("data/landmark_train.csv", "data/landmark_val.csv")
model = build_model(X_train.shape[1], y_train.shape[1])

print("Bắt đầu huấn luyện mô hình...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Lưu mô hình
model.save("models/hand_gesture_model.h5")
print("Mô hình đã được lưu thành công!")
