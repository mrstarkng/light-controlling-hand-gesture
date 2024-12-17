import serial
import time

class LightController:
    def __init__(self, port, baudrate=9600):
        # Initialize the serial connection
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Allow time for the connection to initialize
        print("Serial connection initialized.")

    def send_command(self, command):
        """Send a command to control the light."""
        self.ser.write(f"{command}\n".encode())
        print(f"Sent: {command}")

    def close(self):
        """Close the serial connection."""
        self.ser.close()
        print("Serial connection closed.")
