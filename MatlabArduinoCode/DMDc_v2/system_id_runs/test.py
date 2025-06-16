import pickle
import numpy as np
from pydmd import DMDc

# === Load .pkl data ===
with open('raw_run_20250615_211719_A0_50_Hz5_0.pkl', 'rb') as f:
    data = pickle.load(f)

# === Define state and control variables ===
state_vars = ['positions', 'velocities', 'angles', 'angular_velocities']
control_var = 'commands'  # 1D control input

# === Stack state data ===
X = np.vstack([np.array(data[k]) for k in state_vars])  # (4, N)
U = np.array(data[control_var]).reshape(1, -1)           # (1, N)

# === Align snapshots ===
X1 = X[:, :-1]  # (4, N-1)
X2 = X[:, 1:]   # (4, N-1)
U1 = U[:, :-1]  # (1, N-1)

# === DMDc with explicit control input flag ===
dmdc = DMDc(svd_rank=0, control_inputs=True)
dmdc.fit(X1, U1, X2)

# === Print results ===
print("\n=== Identified System Matrices ===")
print("A matrix:\n", dmdc.A)
print("B matrix:\n", dmdc.B)
