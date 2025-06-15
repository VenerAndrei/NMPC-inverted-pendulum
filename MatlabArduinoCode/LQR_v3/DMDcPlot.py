import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt('data_1.log')  # (N, 5)

# Prepare X, U, X'
X = data[:, :4].T            # (4, N)
U = data[:, 4].reshape(1, -1) # (1, N)
X_prime = X[:, 1:]           # (4, N-1)
X = X[:, :-1]                # (4, N-1)
U = U[:, :-1]                # (1, N-1)

# DMDc: Solve X' ≈ A X + B U
Omega = np.vstack([X, U])
AB = X_prime @ np.linalg.pinv(Omega)
A, B = AB[:, :4], AB[:, 4:].reshape(4, 1)

# Validate
X_prime_pred = A @ X + B @ U
error = np.linalg.norm(X_prime - X_prime_pred, 'fro')
print("Prediction error:", error)

# Time steps (assuming uniform time sampling)
time_steps = np.arange(X.shape[1])

# Create subplots for each state variable
state_labels = ['State 1', 'State 2', 'State 3', 'State 4']
plt.figure(figsize=(12, 8))

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(time_steps, X_prime[i], 'b', label='Real data')
    plt.plot(time_steps, X_prime_pred[i], 'r--', label='DMDc prediction')
    plt.title(state_labels[i])
    plt.xlabel('Time step')
    plt.ylabel('State value')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.suptitle('DMDc Prediction vs Real Data', y=1.02)
plt.show()