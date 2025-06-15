import numpy as np
import math  # Using Python's built-in math module instead
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary, FourierLibrary
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import STLSQ
from sklearn.preprocessing import StandardScaler

# Load data - columns: [x, x_dot, θ, θ_dot, u]
data = np.loadtxt('data_1.log')
X = data[:, :4].T            # (4, N) - [x, x_dot, θ, θ_dot]
U = data[:, 4].reshape(1, -1) # (1, N) - control input

# Simple finite difference with central scheme
diff_method = FiniteDifference(d=1, axis=1, drop_endpoints=True)
X_prime = diff_method._differentiate(X, t=np.arange(X.shape[1]))

# Only trim one point from each end
X = X[:, 1:-1]
U = U[:, 1:-1]
X_prime = X_prime[:, :]  # Already trimmed by FiniteDifference

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.T).T
X_prime_scaled = scaler.transform(X_prime.T).T

# Combine state and control
XU = np.vstack([X_scaled, U]).T

# Create feature library - start simple
library = PolynomialLibrary(degree=2, include_bias=False)

# Conservative optimizer settings
optimizer = STLSQ(
    threshold=0.1,          # Start higher, reduce if needed
    alpha=0.05,            # Increased regularization
    max_iter=20,
    normalize_columns=True
)

# Create and fit SINDy model
model = SINDy(
    optimizer=optimizer,
    feature_library=library,
    feature_names=['x', 'x_dot', 'θ', 'θ_dot', 'u']
)

model.fit(XU, x_dot=X_prime_scaled.T, t=np.arange(XU.shape[0]))

# Print results
print("Discovered equations:")
model.print()

# Evaluate
X_prime_pred = model.predict(XU).T
error = np.linalg.norm(X_prime_scaled - X_prime_pred) / np.linalg.norm(X_prime_scaled)
print(f"\nRelative error: {error:.4f}")

# Manual trigonometric terms if needed
if error > 0.1:
    print("\nAdding manual trigonometric features...")
    θ = X_scaled[2, :]
    trig_features = np.vstack([
        np.sin(θ),
        np.cos(θ),
        np.sin(θ)*np.cos(θ)
    ])
    XU_extended = np.hstack([XU, trig_features.T])
    
    # Update feature names
    extended_names = model.feature_names + ['sin(θ)', 'cos(θ)', 'sin(θ)cos(θ)']
    
    # Refit model
    model.fit(XU_extended, x_dot=X_prime_scaled.T)
    model.feature_names = extended_names
    model.print()