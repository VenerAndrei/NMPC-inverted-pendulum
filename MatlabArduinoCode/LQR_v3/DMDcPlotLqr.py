import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt

# Load and prepare data
data = np.loadtxt('data_1.log')  # (N, 5)
# Convert position (cm→m) and velocity (cm/s→m/s)
data[:, 0] /= 100    # x position (1st column)
data[:, 1] /= 100    # x velocity (2nd column)
X = data[:, :4].T               # (4, N)
U = data[:, 4].reshape(1, -1)   # (1, N)
X_prime = X[:, 1:]              # (4, N-1)
X = X[:, :-1]                   # (4, N-1)
U = U[:, :-1]                   # (1, N-1)

# DMDc identification
Omega = np.vstack([X, U])
AB = X_prime @ np.linalg.pinv(Omega)
A, B = AB[:, :4], AB[:, 4:].reshape(4, 1)

# Verify identification
X_prime_pred = A @ X + B @ U
error = np.linalg.norm(X_prime - X_prime_pred, 'fro')
print(f"DMDc identification error: {error:.4f}")

## LQR Design
# Weighting matrices (adjust these based on your system)
Q = np.diag([100,5,100,1]); # Prioritize important states
R = np.eye(1) * 0.1         # Control effort penalty

# Solve Riccati equation
P = solve_discrete_are(A, B, Q, R)

# Compute LQR gain
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
print("\nLQR gain matrix K:\n", K)

# Verify closed-loop stability
A_cl = A - B @ K
eigvals = np.linalg.eigvals(A_cl)
print("\nClosed-loop eigenvalues:", eigvals)
print("Magnitudes:", np.abs(eigvals))

## Closed-loop simulation
N_sim = 200
x0 = X[:, 0]  # Initial state from data
x_ref = np.zeros(4)  # Reference state (can be changed)

# Storage
X_sim = np.zeros((4, N_sim))
U_sim = np.zeros((1, N_sim))

X_sim[:, 0] = x0

for t in range(N_sim-1):
    # LQR control law (reference tracking)
    U_sim[:, t] = -K @ (X_sim[:, t] - x_ref)
    
    # Apply control to system (using identified model)
    X_sim[:, t+1] = A @ X_sim[:, t] + B @ U_sim[:, t]

    # Add small process noise (optional)
    # X_sim[:, t+1] += np.random.normal(0, 0.01, size=4)

## Visualization
plt.figure(figsize=(14, 10))

# Plot states
state_labels = ['State 1', 'State 2', 'State 3', 'State 4']
for i in range(4):
    plt.subplot(3, 2, i+1)
    plt.plot(X_sim[i], 'b', label='Closed-loop')
    plt.axhline(x_ref[i], color='r', linestyle='--', label='Reference')
    plt.title(state_labels[i])
    plt.ylabel('State value')
    plt.legend()
    plt.grid(True)

# Plot control effort
plt.subplot(3, 2, 5)
plt.plot(U_sim.T, 'g')
plt.title('Control Input')
plt.xlabel('Time step')
plt.ylabel('u(t)')
plt.grid(True)

# Plot phase portrait (example for first two states)
plt.subplot(3, 2, 6)
plt.plot(X_sim[0], X_sim[1], 'b')
plt.plot(x0[0], x0[1], 'ro', label='Initial state')
plt.plot(x_ref[0], x_ref[1], 'rx', markersize=10, label='Target')
plt.title('Phase Portrait (States 1 vs 2)')
plt.xlabel('State 1')
plt.ylabel('State 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.suptitle('DMDc-LQR Closed-Loop Control', y=1.02)
plt.show()