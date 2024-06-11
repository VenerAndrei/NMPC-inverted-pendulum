import numpy as np
from scipy.sparse import csc_matrix
from qpsolvers import solve_qp
import matplotlib.pyplot as plt

# Discrete state-space matrices
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

# Horizon
N = 40  # Prediction horizon

# Simulation time
simulation_time = 15  # seconds
dt = 0.02  # time step
num_steps = int(simulation_time / dt)

# State and control dimensions
nx = A.shape[0]
nu = B.shape[1]

# Construct the extended cost matrices for the QP problem
Q_bar = np.kron(np.eye(N+1), Q)  # Includes an extra block for the initial state
R_bar = np.kron(np.eye(N), R)
H = np.block([
    [Q_bar, np.zeros(((N+1) * nx, N * nu))],
    [np.zeros((N * nu, (N+1) * nx)), R_bar]
])

# Initial state
initial_state = np.array([0.1, 0, 0.1, 0])  # Example initial state

# State trajectory and control input storage
x_trajectory = [initial_state]
u_trajectory = []

# Convert to sparse matrices for performance
H_qp_sparse = csc_matrix(2 * H)  # qpsolvers expects 2*H due to the factor of 1/2 in front of the quadratic term

for step in range(num_steps):
    # Equality constraints matrices
    Ax = np.kron(np.eye(N+1), -np.eye(nx)) + np.kron(np.eye(N+1, k=-1), A)
    Bu = np.kron(np.vstack([np.zeros((1, N)), np.eye(N)]), B)
    Aeq = np.hstack([Ax, Bu])
    Aeq_sparse = csc_matrix(Aeq)

    # Initial state constraint
    current_state = x_trajectory[-1]
    beq = np.zeros((N + 1) * nx)
    beq[:nx] = -current_state

    # Set bounds for states and inputs (for example -10 to 10 for inputs)
    u_min = -10 * np.ones(N * nu)
    u_max = 10 * np.ones(N * nu)
    x_min = -np.inf * np.ones((N + 1) * nx)
    x_max = np.inf * np.ones((N + 1) * nx)

    # Combine state and input bounds
    bounds_min = np.hstack([x_min, u_min])
    bounds_max = np.hstack([x_max, u_max])

    # QP matrices
    f = np.zeros(H_qp_sparse.shape[0])

    # Solve the QP problem using qpsolvers
    x_opt = solve_qp(P=H_qp_sparse, q=f, G=None, h=None, A=Aeq_sparse, b=beq, lb=bounds_min, ub=bounds_max, solver='osqp')

    # Extract the optimal control input
    u_opt = x_opt[(N+1)*nx:(N+1)*nx + nu]
    u_trajectory.append(u_opt)

    # Apply the control input to the system
    next_state = A @ current_state + B @ u_opt
    x_trajectory.append(next_state)

# Convert trajectories to numpy arrays for plotting
x_trajectory = np.array(x_trajectory)
u_trajectory = np.array(u_trajectory)

# Plotting
time = np.linspace(0, simulation_time, num_steps + 1)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(time, x_trajectory[:, 0], label='Position (x)')
plt.plot(time, x_trajectory[:, 1], label='Velocity (x_dot)')
plt.plot(time, x_trajectory[:, 2], label='Angle (theta)')
plt.plot(time, x_trajectory[:, 3], label='Angular Velocity (theta_dot)')
plt.title('State Trajectories')
plt.xlabel('Time (s)')
plt.ylabel('States')
plt.legend()

plt.subplot(2, 1, 2)
plt.step(time[:-1], u_trajectory, where='post')
plt.title('Control Input')
plt.xlabel('Time (s)')
plt.ylabel('Control Input (u)')

plt.tight_layout()
plt.show()
