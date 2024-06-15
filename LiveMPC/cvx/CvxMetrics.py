import numpy as np
import cvxpy as cp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

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

# Initial state and reference
x0 = np.array([0, 0, 0.1, 0])  # initial state [cart position, cart velocity, theta, omega]
xref = np.array([0, 0, 0, 0])  # reference state [cart position, cart velocity, theta, omega]

# Prediction horizons and solvers to test
prediction_horizons = [10, 20, 30, 40, 50]
solvers = [cp.OSQP, cp.SCS]  # Removed ECOS
solverNames= ["OSQP","SCS"]
solvNo = -1;

# Storage for results
results = []
# Run simulations
for solver in solvers:
    solvNo += 1
    solver_name = solverNames[solvNo] # Get the solver class name
    for horizon in prediction_horizons:
        nx = A.shape[0]  # number of states
        nu = B.shape[1]  # number of inputs

        x = cp.Variable((nx, horizon + 1))
        u = cp.Variable((nu, horizon))
        x_init = cp.Parameter(nx)
        x_ref = cp.Parameter(nx)

        cost = 0
        constraints = [x[:, 0] == x_init]
        for t in range(horizon):
            cost += cp.quad_form(x[:, t] - x_ref, Q) + cp.quad_form(u[:, t], R)
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]
            constraints += [cp.abs(u[:, t]) <= 10]  # control input constraints

        prob = cp.Problem(cp.Minimize(cost), constraints)

        x_hist = np.zeros((nx, N + 1))
        u_hist = np.zeros((nu, N))
        x_hist[:, 0] = x0

        times = []

        for i in range(N):
            x_init.value = x_hist[:, i]
            x_ref.value = xref

            start_time = time.time()
            try:
                prob.solve(solver=solver, verbose=False)
                solve_time = time.time() - start_time
                times.append(solve_time)
                
                if prob.status != cp.OPTIMAL:
                    print(f"Solver {solver_name} failed to find an optimal solution at time step {i}.")
                    u_hist[:, i] = 0
                else:
                    u_hist[:, i] = u.value[:, 0]

            except cp.error.SolverError as e:
                print(f"Solver {solver_name} failed at time step {i} with error: {e}")
                u_hist[:, i] = 0

            x_hist[:, i + 1] = A @ x_hist[:, i] + B @ u_hist[:, i]

        min_time = np.min(times)
        max_time = np.max(times)
        avg_time = np.mean(times)

        results.append((solver_name, horizon, min_time, max_time, avg_time))

# Print results
print(f"{'Solver':<10} {'Horizon':<10} {'Min Time':<15} {'Max Time':<15} {'Avg Time':<15}")
for result in results:
    solver_name, horizon, min_time, max_time, avg_time = result
    print(f"{solver_name:<10} {horizon:<10} {min_time:<15.6f} {max_time:<15.6f} {avg_time:<15.6f}")

# Plotting results (optional)
for solver_name, horizon, min_time, max_time, avg_time in results:
    plt.plot(horizon, avg_time, 'o', label=f'{solver_name} Avg Time')

plt.xlabel('Prediction Horizon')
plt.ylabel('Average Solve Time (s)')
plt.title('Solver Performance Comparison')
plt.show()
