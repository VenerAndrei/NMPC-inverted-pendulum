
import serial
import struct
import time

# Serial port configuration
SERIAL_PORT = 'COM5'  # Adjust as needed
BAUD_RATE = 500000
STRUCT_FORMAT = '<7f'  # float32 x 7
SAMPLE_SIZE = struct.calcsize(STRUCT_FORMAT)
TIMEOUT = 2

def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    time.sleep(2)
    print(f"Listening on {SERIAL_PORT} at {BAUD_RATE} baud...")

    prev_time = time.time()

    try:
        while True:
            if ser.in_waiting >= SAMPLE_SIZE:
                raw = ser.read(SAMPLE_SIZE)
                values = struct.unpack(STRUCT_FORMAT, raw)
                now = time.time()
                dt = now - prev_time
                prev_time = now

                print(f"Δt = {dt*1000:.2f} ms | "
                      f"Pos: {values[0]:.3f}, Vel: {values[1]:.3f}, "
                      f"Angle: {values[2]:.3f}, AngVel: {values[3]:.3f}, "
                      f"Cmd: {values[4]:.3f}, Chirp: {values[5]:.3f}, Pot: {values[6]:.3f}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
