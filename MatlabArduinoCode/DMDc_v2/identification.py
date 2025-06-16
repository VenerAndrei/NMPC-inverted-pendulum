import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def perform_dmdc(X1, X2, U):
    XU = np.vstack((X1, U))
    AB = X2 @ np.linalg.pinv(XU)
    n = X1.shape[0]
    A = AB[:, :n]
    B = AB[:, n:]
    return A, B

def simulate_dmdc(A, B, x0, U):
    X_pred = [x0]
    x = x0
    for u in U.T:
        x = A @ x + B @ u
        X_pred.append(x)
    return np.array(X_pred).T  # (states x time)

# --- CONFIG ---
DATA_DIR = "system_id_runs"
start_idx = 10  # Discard first 10 samples
filepaths = glob(os.path.join(DATA_DIR, "*.pkl"))

if not filepaths:
    print("❌ No .pkl files found in", DATA_DIR)
    exit()

# --- Use first file for plotting ---
filepath = filepaths[0]
print(f"\n📂 Processing: {os.path.basename(filepath)}")
with open(filepath, 'rb') as f:
    data = pickle.load(f)

timestamps = np.array(data['timestamps'])[start_idx:]

# --- Convert to SI units ---
pos = np.array(data['positions'])[start_idx:] / 100.0
vel = np.array(data['velocities'])[start_idx:] / 100.0
ang = np.array(data['angles'])[start_idx:]
ang_vel = np.array(data['angular_velocities'])[start_idx:]
command = np.array(data['commands'])[start_idx:]                    # m/s²
chirp = np.array(data['chirp_signals'])[start_idx:] / 100.0        # cm/s² → m/s²

# --- Use the sum of command + chirp as effective control input ---
u_total = command + chirp

# --- Stack states and inputs ---
X = np.vstack([pos, vel, ang, ang_vel])
X1 = X[:, :-1]
X2 = X[:, 1:]
U = u_total[:-1].reshape(1, -1)  # Shape: (1 x N)

# --- Run DMDc ---
A, B = perform_dmdc(X1, X2, U)

# --- Simulate using DMDc model ---
x0 = X[:, 0]
X_sim = simulate_dmdc(A, B, x0, U)

# --- Plot ---
labels = ["Position (m)", "Velocity (m/s)", "Angle (rad)", "Angular Velocity (rad/s)"]
time = timestamps[:X_sim.shape[1]]

plt.figure(figsize=(12, 10))
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(time, X[i, :X_sim.shape[1]], label="Actual", linewidth=2)
    plt.plot(time, X_sim[i], '--', label="DMDc Prediction", linewidth=2)
    plt.ylabel(labels[i])
    plt.grid(True)
    if i == 0:
        plt.legend()
plt.xlabel("Time (s)")
plt.tight_layout()
# plt.suptitle("DMDc Prediction vs Real Data\n(Using Effective Input = Command + Chirp)", fontsize=16, y=1.02)
plt.show()

print(A)
print(B)