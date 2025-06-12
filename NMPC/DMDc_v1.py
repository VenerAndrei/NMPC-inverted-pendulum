import numpy as np
import serial
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pydmd import DMDc

# ------------------- CONFIG -------------------
SERIAL_PORT = 'COM5'       # Adjust for your platform
BAUD_RATE = 115200
SAMPLING_RATE = 50          # Hz
DURATION = 30               # seconds
NUM_SAMPLES = SAMPLING_RATE * DURATION
CHIRP_START_FREQ = 0.1      # Hz
CHIRP_END_FREQ = 5.0        # Hz
CHIRP_AMPLITUDE = 1.0       # Units consistent with actuator
SEND_TRIGGER = b'i'         # Command to start chirp and send data
# ----------------------------------------------

def generate_chirp_signal():
    t = np.linspace(0, DURATION, NUM_SAMPLES)
    chirp = CHIRP_AMPLITUDE * np.sin(2 * np.pi * t * (CHIRP_START_FREQ + (CHIRP_END_FREQ - CHIRP_START_FREQ) * t / DURATION))
    return t, chirp

def collect_data(serial_conn, chirp_signal):
    state_data = []
    control_data = []
    timestamps = []

    print("[INFO] Starting data collection...")
    serial_conn.write(SEND_TRIGGER)
    time.sleep(0.5)

    start_time = time.time()
    while len(state_data) < NUM_SAMPLES:
        try:
            line = serial_conn.readline().decode().strip()
            if not line:
                continue
            parts = list(map(float, line.split('\t')))
            if len(parts) != 5:
                continue

            state_data.append(parts[:4])
            control_data.append(parts[4])
            timestamps.append(time.time() - start_time)

            if len(state_data) % 100 == 0:
                print(f"[INFO] Samples collected: {len(state_data)}/{NUM_SAMPLES}")
        except Exception as e:
            print(f"[WARNING] Failed to parse line: {e}")
            continue

    print("[INFO] Data collection complete.")
    return np.array(timestamps), np.array(state_data).T, np.array(control_data).reshape(1, -1)

def perform_dmdc(X, U, dt, rank=4):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.T).T
    U_trimmed = U[:, :-1]

    dmdc = DMDc(svd_rank=rank)
    dmdc.fit(X_scaled[:, :-1], X_scaled[:, 1:], U_trimmed)

    print("[RESULT] Identified system matrix A:")
    print(dmdc.A)
    print("\n[RESULT] Identified input matrix B:")
    print(dmdc.B)

    return dmdc

def main():
    t, chirp_signal = generate_chirp_signal()
    
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        input("[READY] Press ENTER to start data collection...")
        timestamps, X, U = collect_data(ser, chirp_signal)

    dt = np.mean(np.diff(timestamps))
    dmdc_model = perform_dmdc(X, U, dt, rank=4)

    # Save results
    with open("dmdc_results.pkl", "wb") as f:
        pickle.dump({
            'A': dmdc_model.A,
            'B': dmdc_model.B,
            'scaler': dmdc_model.scaler,
            'timestamps': timestamps,
            'X': X,
            'U': U
        }, f)
        print("[INFO] Results saved to dmdc_results.pkl")

    # Plot example state and control signal
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(['Pos', 'Vel', 'Angle', 'Ang Vel']):
        plt.subplot(5, 1, i+1)
        plt.plot(timestamps, X[i])
        plt.ylabel(label)
        plt.grid(True)
    plt.subplot(5, 1, 5)
    plt.plot(timestamps, U[0])
    plt.ylabel("Input")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
