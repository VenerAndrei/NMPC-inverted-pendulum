import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
DATA_DIR = "system_id_runs"
FILENAME = input("Enter filename (just the name, not full path): ").strip()
filepath = os.path.join(DATA_DIR, FILENAME)

# --- LOAD ---
if not os.path.exists(filepath):
    print(f"❌ File not found: {filepath}")
    exit()

with open(filepath, 'rb') as f:
    data = pickle.load(f)

timestamps = np.array(data['timestamps'])
signals = {
    "Position (cm)": np.array(data['positions']),
    "Velocity (cm/s)": np.array(data['velocities']),
    "Angle (rad)": np.array(data['angles']),
    "Angular Velocity (rad/s)": np.array(data['angular_velocities']),
    "Command (m/s²)": np.array(data['commands']),
    "Chirp Input (cm/s²)": np.array(data['chirp_signals']),
}

# --- PLOT ---
plt.figure(figsize=(12, 10))
for i, (label, signal) in enumerate(signals.items(), start=1):
    plt.subplot(len(signals), 1, i)
    plt.plot(timestamps, signal, linewidth=1.5)
    plt.grid(True)
    plt.ylabel(label)
    if i == len(signals):
        plt.xlabel("Time (s)")

plt.tight_layout()
plt.suptitle("Saved Run Data", fontsize=16, y=1.02)
plt.show()
