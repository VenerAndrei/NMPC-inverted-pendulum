import numpy as np
import pandas as pd
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary, FourierLibrary
from pysindy.optimizers import STLSQ
import matplotlib.pyplot as plt

# Load the log file into a DataFrame
data = pd.read_csv("log.txt", delimiter="\t", header=None)
data.columns = ["time", "pos", "velocity", "theta", "dtheta", "stepper_velocity"]

# Filter out rows where stepper_velocity is 0 or 10
filtered_data = data[(data["stepper_velocity"] != 0) & (data["stepper_velocity"] != 10) & (data["stepper_velocity"] != -10)]

# Drop the time column
filtered_data = filtered_data.drop(columns=["time"])

# Extract state data (theta, dtheta, pos, velocity)
X = filtered_data[["pos", "velocity", "theta", "dtheta"]].values

# Extract control input (stepper_velocity)
u = filtered_data["stepper_velocity"].values

# Define the feature library (e.g., polynomials and trigonometric functions)
feature_library = PolynomialLibrary(degree=4) + FourierLibrary(n_frequencies=2)

# Initialize the SINDy model
model = SINDy(
    feature_library=feature_library,
    optimizer=STLSQ(threshold=0.001),  # Sparse Thresholded Least Squares
    feature_names=["pos", "velocity", "theta", "dtheta"],
)

# Fit the model to the data
model.fit(X, t=None, u=u)

# Print the identified equations
print("Identified equations:")
model.print()

# Simulate the model forward in time
X_sim = model.simulate(X[0], t=len(X), u=u)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X[:, 0], label="True theta")
plt.plot(X_sim[:, 0], "--", label="Simulated theta")
plt.xlabel("Time steps")
plt.ylabel("Theta")
plt.legend()
plt.title("True vs Simulated Theta")
plt.show()