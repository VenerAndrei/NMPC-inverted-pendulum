import numpy as np

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

print("A (dynamics):\n", A)
print("\nB (control):\n", B)

# Validate
X_prime_pred = A @ X + B @ U
error = np.linalg.norm(X_prime - X_prime_pred, 'fro')
print("\nPrediction error:", error)