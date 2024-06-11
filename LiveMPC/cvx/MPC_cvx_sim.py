import numpy as np
import cvxpy as cp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the system matrices and MPC parameters
A = np.array([
    [1, 0.02, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1.004, 0.02002],
    [0, 0, 0.3579, 1.004]
])

B = np.array([
    [0.0002],
    [0.02],
    [-0.000365],
    [-0.03652]
])

Q = np.diag([20, 1, 1, 1])
R = np.eye(1)

# Time parameters
dt = 0.02   # time step (s)
T = 10      # total time (s)
N = int(T/dt)  # number of time steps
horizon = 50  # prediction horizon

# Initial state and reference
x0 = np.array([0, 0, 0.1, 0])      # initial state [cart position, cart velocity, theta, omega,]
xref = np.array([0, 0, 0, 0])      # reference state [cart position, cart velocity, theta, omega]

# MPC setup
nx = A.shape[0]  # number of states
nu = B.shape[1]  # number of inputs

x = cp.Variable((nx, horizon+1))
u = cp.Variable((nu, horizon))
x_init = cp.Parameter(nx)
x_ref = cp.Parameter(nx)

cost = 0
constraints = [x[:, 0] == x_init]
for t in range(horizon):
    cost += cp.quad_form(x[:, t] - x_ref, Q) + cp.quad_form(u[:, t], R)
    constraints += [x[:, t+1] == A @ x[:, t] + B @ u[:, t]]
    constraints += [cp.abs(u[:, t]) <= 10]  # control input constraints

prob = cp.Problem(cp.Minimize(cost), constraints)

# Storage for results
x_hist = np.zeros((nx, N+1))
u_hist = np.zeros((nu, N))

x_hist[:, 0] = x0

# MPC loop
for i in range(N):
    x_init.value = x_hist[:, i]
    x_ref.value = xref
    prob.solve(solver=cp.OSQP)

    u_hist[:, i] = u.value[:, 0]
    x_hist[:, i+1] = A @ x_hist[:, i] + B @ u_hist[:, i]

# Plot results
t = np.linspace(0, T, N+1)
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(t, x_hist[2, :], label='theta (rad)')
plt.ylabel('Theta (rad)')
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(t, x_hist[3, :], label='omega (rad/s)')
plt.ylabel('Omega (rad/s)')
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(t, x_hist[1, :], label='position (m)')
plt.ylabel('Position (m)')
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(t, x_hist[2, :], label='velocity (m/s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.step(t[:-1], u_hist[0, :], where='post', label='u (N)')
plt.xlabel('Time (s)')
plt.ylabel('Control input (N)')
plt.legend()
plt.show()
