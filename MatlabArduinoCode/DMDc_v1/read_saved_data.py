import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Replace this with your actual file path
DATA_FILE = "system_id_runs/raw_run_20250615_183001.pkl"

def load_data(file_path):
    """Load sensor data from a saved .pkl file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print("\nMetadata:")
    for key, value in data['metadata'].items():
        print(f"{key}: {value}")

    return data

def plot_data(data):
    """Plot all sensor data"""
    t = np.array(data['timestamps'])
    signals = [
        ("Position (cm)", data['positions']),
        ("Velocity (cm/s)", data['velocities']),
        ("Angle (rad)", data['angles']),
        ("Angular Velocity (rad/s)", data['angular_velocities']),
        ("Chirp Input", data['chirp_signals'])
    ]

    plt.figure(figsize=(12, 10))
    for i, (label, signal) in enumerate(signals):
        plt.subplot(len(signals), 1, i + 1)
        plt.plot(t, signal, linewidth=1.5)
        plt.ylabel(label)
        plt.grid(True)
        if i == len(signals) - 1:
            plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        data = load_data(DATA_FILE)
        plot_data(data)
    except Exception as e:
        print(f"Error: {e}")
