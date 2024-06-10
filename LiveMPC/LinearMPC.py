import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Discrete state-space matrices from Matlab
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

C = np.eye(4)
D = np.zeros((4, 1))
Q = np.diag([100, 1, 100, 1])
R = np.eye(1)

# Constraints
lower_bounds = -10
upper_bounds = 10

# MPC parameters
N_pred = 50  # prediction horizon
N_sim = int(3 / 0.02)  # simulate first 3 seconds
x_initial = np.array([0, 0, 0.1, 0])  # initial state with small angle

# CasADi setup
opti = ca.Opti()

# Decision variables
x = opti.variable(4, N_pred + 1)
u = opti.variable(1, N_pred)

# Parameters
xinit = opti.parameter(4, 1)
uinit = opti.parameter(1, 1)
yref = opti.parameter(4, N_pred + 1)

# Objective function
objective = 0
for k in range(N_pred):
    if k > 0:
        objective += ca.mtimes((C @ x[:, k] - yref[:, k]).T, Q @ (C @ x[:, k] - yref[:, k])) + ca.mtimes((u[:, k] - u[:, k - 1]).T, R @ (u[:, k] - u[:, k - 1]))
    else:
        objective += ca.mtimes((C @ x[:, k] - yref[:, k]).T, Q @ (C @ x[:, k] - yref[:, k])) + ca.mtimes((u[:, k] - uinit).T, R @ (u[:, k] - uinit))
objective += ca.mtimes((C @ x[:, N_pred] - yref[:, N_pred]).T, Q @ (C @ x[:, N_pred] - yref[:, N_pred]))

opti.minimize(objective)

# System dynamics and constraints
opti.subject_to(x[:, 0] == xinit)
for k in range(N_pred):
    opti.subject_to(x[:, k + 1] == A @ x[:, k] + B @ u[:, k])
    opti.subject_to(lower_bounds <= u[:, k])
    opti.subject_to(u[:, k] <= upper_bounds)

# Solver settings
opts = {
    "ipopt.print_level": 0,
    "print_time": 0,
    "ipopt.sb": 'yes'
}
opti.solver('ipopt', opts)

# Simulation loop
u_sim = np.zeros((1, N_sim))
x_sim = np.zeros((4, N_sim + 1))
x_sim[:, 0] = x_initial

for i in range(N_sim):
    if i == 0:
        u_init = np.zeros((1, 1))
    else:
        u_init = u_sim[:, i - 1].reshape(-1, 1)

    opti.set_value(xinit, x_sim[:, i])
    opti.set_value(uinit, u_init)

    # Set yref to zero reference
    opti.set_value(yref, np.zeros((4, N_pred + 1)))

    sol = opti.solve()
    u_sim[:, i] = sol.value(u[:, 0])
    x_sim[:, i + 1] = A @ x_sim[:, i] + B @ u_sim[:, i]

# Plotting results
time = np.arange(0, N_sim) * 0.02
plt.figure(figsize=(10, 6))
plt.plot(time, x_sim[0, :-1], 'r', label='x')
plt.plot(time, x_sim[1, :-1], 'g', label='dx')
plt.plot(time, x_sim[2, :-1], 'b', label='theta')
plt.plot(time, x_sim[3, :-1], 'm', label='dtheta')
plt.xlabel('Time (s)')
plt.ylabel('State values')
plt.title('MPC Control - All States')
plt.legend()
plt.grid(True)
plt.show()

