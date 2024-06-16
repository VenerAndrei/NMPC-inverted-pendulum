import numpy as np
import cvxpy as cp
import casadi as ca
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
solvers = [cp.OSQP, cp.SCS]
solverNames = ["OSQP", "SCS"]

# Storage for results
results = []

# Run simulations for CVXPY solvers
for solver, solver_name in zip(solvers, solverNames):
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

# CasADi simulation
def casadi_mpc_simulation(N_pred, N_sim, x_initial, A, B, Q, R, lower_bounds, upper_bounds):
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
            objective += ca.mtimes((x[:, k] - yref[:, k]).T, Q @ (x[:, k] - yref[:, k])) + ca.mtimes((u[:, k] - u[:, k - 1]).T, R @ (u[:, k] - u[:, k - 1]))
        else:
            objective += ca.mtimes((x[:, k] - yref[:, k]).T, Q @ (x[:, k] - yref[:, k])) + ca.mtimes((u[:, k] - uinit).T, R @ (u[:, k] - uinit))
    objective += ca.mtimes((x[:, N_pred] - yref[:, N_pred]).T, Q @ (x[:, N_pred] - yref[:, N_pred]))

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

    times = []

    for i in range(N_sim):
        if i == 0:
            u_init = np.zeros((1, 1))
        else:
            u_init = u_sim[:, i - 1].reshape(-1, 1)

        opti.set_value(xinit, x_sim[:, i])
        opti.set_value(uinit, u_init)

        # Set yref to zero reference
        opti.set_value(yref, np.zeros((4, N_pred + 1)))

        start_time = time.time()
        sol = opti.solve()
        solve_time = time.time() - start_time
        times.append(solve_time)
        
        u_sim[:, i] = sol.value(u[:, 0])
        x_sim[:, i + 1] = A @ x_sim[:, i] + B @ u_sim[:, i]

    min_time = np.min(times)
    max_time = np.max(times)
    avg_time = np.mean(times)

    return N_pred, min_time, max_time, avg_time

# Run CasADi simulation
N_pred = 50  # prediction horizon
N_sim = int(T / dt)  # simulate for T seconds
x_initial = x0

casadi_result = casadi_mpc_simulation(N_pred, N_sim, x_initial, A, B, Q, R, -10, 10)
results.append(('CasADi', *casadi_result))

# Print results
print(f"{'Solver':<10} {'Horizon':<10} {'Min Time':<15} {'Max Time':<15} {'Avg Time':<15}")
for result in results:
    solver_name, horizon, min_time, max_time, avg_time = result
    print(f"{solver_name:<10} {horizon:<10} {min_time:<15.6f} {max_time:<15.6f} {avg_time:<15.6f}")

# Plotting results
colors = {"OSQP": "red", "SCS": "blue", "CasADi": "green"}
markers = {"OSQP": "o", "SCS": "s", "CasADi": "d"}

plt.figure(figsize=(10, 6))
for solver_name in solverNames + ["CasADi"]:
    horizons = [res[1] for res in results if res[0] == solver_name]
    avg_times = [res[4] for res in results if res[0] == solver_name]
    plt.plot(horizons, avg_times, marker=markers[solver_name], color=colors[solver_name], label=f'{solver_name} Avg Time')

plt.xlabel('Prediction Horizon')
plt.ylabel('Average Solve Time (s)')
plt.title('Solver Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()
