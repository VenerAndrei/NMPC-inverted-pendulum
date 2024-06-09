import serial
from datetime import datetime

# Open the serial port
ser = serial.Serial('COM4', 921600)  # Replace 'COM4' with your actual COM port
if not ser.is_open:
    ser.open()

try:
    print("Reading data from the serial port...")
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds
            print(f"{current_time} - {line}")
            ser.write(b'CONTINUE\n')  # Send the continue command

            
except KeyboardInterrupt:
    print("Exiting program.")
finally:
    ser.close()
