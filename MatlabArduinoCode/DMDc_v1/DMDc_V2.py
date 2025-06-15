import numpy as np
import serial
import matplotlib.pyplot as plt
from pydmd import DMDc
from sklearn.preprocessing import StandardScaler
import time
import keyboard
import os
from datetime import datetime
import pickle

# Serial port configuration
SERIAL_PORT = 'COM5'  # Change as needed
BAUD_RATE = 500000
TIMEOUT = 1

# Data collection parameters
CHIRP_DURATION = 30  # seconds
SAMPLING_RATE = 50   # Hz
TOTAL_SAMPLES = int(CHIRP_DURATION * SAMPLING_RATE)

DATA_DIR = "system_id_runs"
os.makedirs(DATA_DIR, exist_ok=True)

def parse_struct_line(line):
    try:
        parts = [float(x) for x in line.strip().split('\t')]
        if len(parts) == 5:
            return tuple(parts)  # pos, vel, angle, angle_vel, chirp
    except ValueError:
        pass
    return None

def save_raw_data(timestamps, positions, velocities, angles, angular_velocities, chirp_signals, metadata):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{DATA_DIR}/raw_run_{timestamp_str}.pkl"
    data_dict = {
        'timestamps': timestamps,
        'positions': positions,
        'velocities': velocities,
        'angles': angles,
        'angular_velocities': angular_velocities,
        'chirp_signals': chirp_signals,
        'metadata': metadata
    }
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"\nRaw data saved to {filename}")
    return filename

def collect_data_with_manual_trigger():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    time.sleep(2)

    timestamps, positions, velocities, angles, angular_velocities, chirp_signals = [], [], [], [], [], []

    print("\n" + "="*50)
    print("DMDc V2: Reading struct data from PendulV2")
    print("="*50)
    print("Press 's' to start data collection, 'q' to quit")

    collecting = False
    start_time = None
    metadata = {
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'description': ''
    }

    try:
        while True:
            if keyboard.is_pressed('s') and not collecting:
                ser.write(b'i')
                print("\nData collection started!")
                collecting = True
                start_time = time.time()
                metadata['chirp_start'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metadata['description'] = input("Enter description for this run: ")

            if keyboard.is_pressed('q'):
                print("\nUser requested stop.")
                break

            if ser.in_waiting:
                line = ser.readline().decode('utf-8').strip()
                parsed = parse_struct_line(line)
                if parsed and collecting:
                    pos, vel, ang, ang_vel, chirp = parsed
                    timestamps.append(time.time() - start_time)
                    positions.append(pos)
                    velocities.append(vel)
                    angles.append(ang)
                    angular_velocities.append(ang_vel)
                    chirp_signals.append(chirp)

                    if len(timestamps) % 50 == 0:
                        print(f"Collected {len(timestamps)}/{TOTAL_SAMPLES} samples")

                    if len(timestamps) >= TOTAL_SAMPLES:
                        print("\nFinished data collection.")
                        break
    except KeyboardInterrupt:
        print("\nKeyboard interrupt")
    finally:
        ser.close()
        metadata['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata['samples_collected'] = len(timestamps)

        if timestamps:
            return save_raw_data(
                np.array(timestamps),
                np.array(positions),
                np.array(velocities),
                np.array(angles),
                np.array(angular_velocities),
                np.array(chirp_signals),
                metadata
            ), metadata
        else:
            print("No data collected.")
            return None, None

def main():
    raw_filename, _ = collect_data_with_manual_trigger()
    if raw_filename:
        print("\nRun saved. You can process this with RunIdentification.py")

if __name__ == "__main__":
    main()
