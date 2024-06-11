import serial
from datetime import datetime
import numpy as np
import cvxpy as cp
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
Q = np.diag([200, 1, 100, 1])
R = np.diag([10])  # Increased the weight on control input to penalize large values

# MPC setup
horizon = 20  # prediction horizon
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
    constraints += [cp.abs(u[:, t]) <= 10]  # Adjusted control input constraints

prob = cp.Problem(cp.Minimize(cost), constraints)

# Serial communication setup
ser = serial.Serial('COM4', 921600)  # Replace 'COM4' with your actual COM port

if not ser.is_open:
    ser.open()

last_uinit = 0
x_ref.value = np.array([0, 0, 0, 0])  # Reference state

try:
    print("Reading data from the serial port...")
    while True:
        if ser.in_waiting > 0:
            # Parse the line into a list of floats
            line = ser.readline().decode('utf-8').strip()
            state_data = np.array([float(x) for x in line.split('\t')])

            # Set the initial state
            x_init.value = state_data

            # Solve the MPC problem
            prob.solve(solver=cp.OSQP)

            # Get the optimal control input
            u_value = u.value[0, 0]

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
import serial
from datetime import datetime
import numpy as np
import cvxpy as cp
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
Q = np.diag([200, 1, 100, 1])
R = np.diag([10])  # Increased the weight on control input to penalize large values

# MPC setup
horizon = 20  # prediction horizon
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
    constraints += [cp.abs(u[:, t]) <= 10]  # Adjusted control input constraints

prob = cp.Problem(cp.Minimize(cost), constraints)

# Serial communication setup
ser = serial.Serial('COM4', 921600)  # Replace 'COM4' with your actual COM port

if not ser.is_open:
    ser.open()

last_uinit = 0
x_ref.value = np.array([0, 0, 0, 0])  # Reference state

try:
    print("Reading data from the serial port...")
    while True:
        if ser.in_waiting > 0:
            # Parse the line into a list of floats
            line = ser.readline().decode('utf-8').strip()
            state_data = np.array([float(x) for x in line.split('\t')])

            # Set the initial state
            x_init.value = state_data

            # Solve the MPC problem
            prob.solve(solver=cp.OSQP)

            # Get the optimal control input
            u_value = u.value[0, 0]

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
import serial
from datetime import datetime
import numpy as np
import cvxpy as cp
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
Q = np.diag([10, 1, 1, 1])
R = np.diag([1])  # Increased the weight on control input to penalize large values

# MPC setup
horizon = 60  # prediction horizon
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
    constraints += [cp.abs(u[:, t]) <= 15]  # Adjusted control input constraints

prob = cp.Problem(cp.Minimize(cost), constraints)

# Serial communication setup
ser = serial.Serial('COM4', 921600)  # Replace 'COM4' with your actual COM port

if not ser.is_open:
    ser.open()

last_uinit = 0
x_ref.value = np.array([0, 0, 0, 0])  # Reference state

try:
    print("Reading data from the serial port...")
    while True:
        if ser.in_waiting > 0:
            # Parse the line into a list of floats
            line = ser.readline().decode('utf-8').strip()
            state_data = np.array([float(x) for x in line.split('\t')])

            # Set the initial state
            x_init.value = state_data

            # Solve the MPC problem
            prob.solve(solver=cp.OSQP)

            # Get the optimal control input
            u_value = u.value[0, 0]

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
