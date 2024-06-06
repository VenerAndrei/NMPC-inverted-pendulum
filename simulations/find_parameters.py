import casadi
import matplotlib.pyplot as plt
import numpy as np

def dynamic(x_state, u, params):
    # x = [x dx th dth];
    # System constants
    g = 9.81
    L = params['L']
    l = L / 2
    m = params['m']
    M = 1.0

    # States
    x = x_state[0]
    dx = x_state[1]
    th = x_state[2]
    dth = x_state[3]

    q_th = params['q_th']
    Mf = q_th * dth

    q_x = 0.001
    Ff = -q_x * dx

    f = u[0]
    ddx = (m*g*casadi.sin(th)*casadi.cos(th)  - (7/3)*(f + m*l*dth*dth*casadi.sin(th) + Ff) - (Mf*casadi.cos(th))/l)/(m*casadi.cos(th)*casadi.cos(th) - (7/3)*M)
    ddx = 0;
    ddth = (g*casadi.sin(th) - ddx*casadi.cos(th) - Mf/(m*l))/((7/3)*l)

    return casadi.vertcat(dx, ddx, dth, ddth)

def rk4(ode, h, x, u, params):
    k1 = ode(x, u, params)
    k2 = ode(x + h/2 * k1, u, params)
    k3 = ode(x + h/2 * k2, u, params)
    k4 = ode(x + h * k3, u, params)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)

# Read log file data
log_file = './ArduinoOutput.log'
log_data = np.loadtxt(log_file, delimiter='\t', skiprows=1)
log_time = np.arange(0, log_data.shape[0] * 0.02, 0.02)

# Define the optimization problem
opti = casadi.Opti()

# Parameters to optimize
L = opti.variable()
m = opti.variable()
q_th = opti.variable()

# Initial guesses
opti.set_initial(L, 1.0)
opti.set_initial(m, 0.1)
opti.set_initial(q_th, 0.005)

# Simulation parameters
t_0 = 0. # start time
t_end = 60. # end time
h = 0.02 # step size
t_arr = np.arange(t_0, t_end, h) # array of integration times
n_steps = len(t_arr) # number of steps

# Initial state
x_0 = np.array([0, 0, -np.pi/2, 0])

# Objective function
error = 0

# Simulation loop
x = casadi.vertcat(*x_0)
u = [0]

for step in range(n_steps):
    x = rk4(dynamic, h, x, u, {'L': L, 'm': m, 'q_th': q_th})
    if step < len(log_data):
    #    error += casadi.sumsqr(x[0] - log_data[step, 0]) + casadi.sumsqr(x[1] - log_data[step, 1]) + casadi.sumsqr(x[2] - log_data[step, 2]) + casadi.sumsqr(x[3] - log_data[step, 3])
       error +=  (x[2] - log_data[step, 2])*(x[2] - log_data[step, 2]); # + (x[3] - log_data[step, 3])*(x[3] - log_data[step, 3])

# Minimize the error
opti.minimize(error)

# Bounds
opti.subject_to(L > 0.2)
opti.subject_to(L < 0.4)
opti.subject_to(m < 0.2)
opti.subject_to(m > 0.01)
opti.subject_to(q_th < 1)
opti.subject_to(q_th > 0)


# Solver
opti.solver('ipopt')

# Solve the optimization problem
sol = opti.solve()

# Optimal values
L_opt = sol.value(L)
m_opt = sol.value(m)
q_th_opt = sol.value(q_th)

print(f'Optimal Length (L): {L_opt}')
print(f'Optimal Mass (m): {m_opt}')
print(f'Optimal Theta Friction (q_th): {q_th_opt}')

# Run the simulation with the optimized parameters
x = np.zeros((4, n_steps))
x[:,0] = x_0
for step in range(1, n_steps):
    x[:,step] = rk4(dynamic, h, x[:,step-1], u, {'L': L_opt, 'm': m_opt, 'q_th': q_th_opt}).full().flatten()

# Plot the results
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t_arr, x[0,:], label='Simulated Position')
plt.plot(log_time, log_data[:,0], label='Logged Position')
plt.grid(True)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Position')

plt.subplot(2, 1, 2)
plt.plot(t_arr, x[1,:], label='Simulated Velocity')
plt.plot(log_time, log_data[:,1], label='Logged Velocity')
plt.grid(True)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Velocity')

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t_arr, x[2,:], label='Simulated Theta')
plt.plot(log_time, log_data[:,2], label='Logged Theta')
plt.grid(True)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Theta')

plt.subplot(2, 1, 2)
plt.plot(t_arr, x[3,:], label='Simulated Angular Velocity')
plt.plot(log_time, log_data[:,3], label='Logged Angular Velocity')
plt.grid(True)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity')

plt.show()
