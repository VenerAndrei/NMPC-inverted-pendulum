import numpy as np
import serial
import struct
import time
import keyboard
import os
from datetime import datetime
import pickle

# Serial config
SERIAL_PORT = 'COM5'
BAUD_RATE = 500000
TIMEOUT = 2
STRUCT_FORMAT = '<7f'
SAMPLE_SIZE = struct.calcsize(STRUCT_FORMAT)

CHIRP_DURATION = 30
SAMPLING_RATE = 50
TOTAL_SAMPLES = int(CHIRP_DURATION * SAMPLING_RATE)

DATA_DIR = "system_id_runs"
os.makedirs(DATA_DIR, exist_ok=True)

def looks_valid(values):
    return (
        len(values) == 7 and
        all(-1000 < v < 1000 for v in values) and
        not any(np.isnan(v) or np.isinf(v) for v in values)
    )

def sync_to_struct(ser, struct_size):
    print("⏳ Syncing to struct boundary...")
    while True:
        head = ser.read(1)
        packet = head + ser.read(struct_size - 1)
        if len(packet) != struct_size:
            continue
        try:
            values = struct.unpack(STRUCT_FORMAT, packet)
            if looks_valid(values):
                print("✅ Sync complete.")
                return
        except struct.error:
            continue

def save_raw_data(timestamps, positions, velocities, angles, angular_velocities, commands, chirps, pots, metadata):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{DATA_DIR}/raw_run_{timestamp_str}.pkl"

    data = {
        'timestamps': timestamps,
        'positions': positions,
        'velocities': velocities,
        'angles': angles,
        'angular_velocities': angular_velocities,
        'commands': commands,
        'chirp_signals': chirps,
        'potentiometer': pots,
        'metadata': metadata
    }

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n✅ Raw data saved to {filename}")
    return filename

def collect_data_with_binary_struct():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    time.sleep(2)
    ser.reset_input_buffer()

    timestamps = []
    positions = []
    velocities = []
    angles = []
    angular_velocities = []
    commands = []
    chirps = []
    pots = []

    print("\n" + "="*50)
    print("📡 Binary Serial Data Collection")
    print("="*50)
    print("Press 's' to start chirp signal and data collection")
    print("Press 'q' to quit")

    collecting = False
    start_time = None
    metadata = {
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'description': ''
    }

    try:
        while True:
            if keyboard.is_pressed('s') and not collecting:
                ser.reset_input_buffer()
                ser.write(b'i')
                time.sleep(0.1)
                ser.reset_input_buffer()
                print("▶️ Trigger sent. Syncing to packet...")
                sync_to_struct(ser, SAMPLE_SIZE)
                print("✅ Sync complete. Starting data collection.")

                collecting = True
                start_time = time.time()
                metadata['chirp_start'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                desc = input("Enter a brief description: ")
                metadata['description'] = desc

            if keyboard.is_pressed('q'):
                print("\n🛑 User aborted.")
                break

            if ser.in_waiting >= SAMPLE_SIZE:
                raw = ser.read(SAMPLE_SIZE)
                if len(raw) != SAMPLE_SIZE:
                    continue
                try:
                    values = struct.unpack(STRUCT_FORMAT, raw)
                    if not looks_valid(values):
                        continue
                except struct.error:
                    continue

                if collecting:
                    now = time.time() - start_time
                    timestamps.append(now)
                    positions.append(values[0])
                    velocities.append(values[1])
                    angles.append(values[2])
                    angular_velocities.append(values[3])
                    commands.append(values[4])
                    chirps.append(values[5])
                    pots.append(values[6])

                    if len(timestamps) % 50 == 0:
                        print(f"📈 Collected {len(timestamps)}/{TOTAL_SAMPLES}")
                        print(f"  Pos: {values[0]:.3f}, Vel: {values[1]:.3f}, "
                              f"Angle: {values[2]:.3f}, AngVel: {values[3]:.3f}, "
                              f"Cmd: {values[4]:.3f}, Chirp: {values[5]:.3f}, Pot: {values[6]:.3f}")

                    if len(timestamps) >= TOTAL_SAMPLES:
                        print("\n✅ Collection complete.")
                        break

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt.")
    finally:
        ser.close()
        metadata['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata['samples_collected'] = len(timestamps)

        if timestamps:
            print(f"\nℹ️ Duration: {timestamps[-1]:.2f}s, Samples: {len(timestamps)}")
            return save_raw_data(
                np.array(timestamps),
                np.array(positions),
                np.array(velocities),
                np.array(angles),
                np.array(angular_velocities),
                np.array(commands),
                np.array(chirps),
                np.array(pots),
                metadata
            ), metadata
        else:
            print("⚠️ No data collected.")
            return None, None

def main():
    raw_filename, _ = collect_data_with_binary_struct()
    if not raw_filename:
        return
    print(f"\nTo analyze this run, use:\n  perform_identification('{raw_filename}')")

if __name__ == "__main__":
    main()
