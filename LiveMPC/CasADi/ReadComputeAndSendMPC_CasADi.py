import serial
from datetime import datetime
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Discrete state-space matrices provided
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
lower_bounds = -25
upper_bounds = 25

# MPC parameters
N_pred = 20  # prediction horizon
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

# Open the serial port
ser = serial.Serial('COM4', 921600)  # Replace 'COM4' with your actual COM port

if not ser.is_open:
    ser.open()

last_uinit = 0;
try:
    print("Reading data from the serial port...")
    while True:
        if ser.in_waiting > 0:
            
            # Parse the line into a list of floats
            line = ser.readline().decode('utf-8').strip()
            state_data = np.array([float(x) for x in line.split('\t')])

            # Set the current state value for the optimizer
            opti.set_value(xinit, state_data)
            opti.set_value(uinit, last_uinit)
            opti.set_value(yref, np.zeros((4, N_pred + 1)))

            # Solve the MPC problem
            sol = opti.solve()
            u_value = sol.value(u[:, 0])
            last_uinit = u_value

            # Print the timestamp, state, and control input on the same line
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds
            state_str = '\t'.join([f"{s:.2f}" for s in state_data])
            print(f"{current_time} - State: {state_str} - Command: {u_value:.2f}")

            # Send the control input back to the Arduino
            ser.write(f"{u_value}\n".encode('utf-8'))
         
except KeyboardInterrupt:
    print("Exiting program.")
finally:
    ser.close()
