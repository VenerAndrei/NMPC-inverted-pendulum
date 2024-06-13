import serial
from datetime import datetime
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import signal
import sys

# q = 0.01
# A = np.array([
#     [1, 0.02, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1.006, 0.01971],
#     [0, 0, 0.552, 0.9734]
# ])

# B = np.array([
#     [0.0002],
#     [0.02],
#     [-0.0005658],
#     [-0.05633]
# ])
A = np.array([
    [1, 0.02, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1.006, 0.02],
    [0, 0, 0.5601, 1.002]
])

B = np.array([
    [0.0002],
    [0.02],
    [-0.0005713],
    [-0.05716]
])

# # PERCUTEAZA BINE CU ASTEA
# C = np.eye(4)
# D = np.zeros((4, 1))
# Q = np.diag([1, 1, 200, 1])
# R = np.diag([9])  # Increased the weight on control input to penalize large values
# P = 1*Q  # Terminal cost matrix (can be set equal to Q or another matrix)

C = np.eye(4)
D = np.zeros((4, 1))
Q = np.diag([40, 1, 10, 1])
R = np.diag([0.9])  # Increased the weight on control input to penalize large values
gamma = 1
P = 1*Q  # Terminal cost matrix (can be set equal to Q or another matrix)

# MPC setup
horizon = 50  # prediction horizon
nx = A.shape[0]  # number of states
nu = B.shape[1]  # number of inputs

x = cp.Variable((nx, horizon+1))
u = cp.Variable((nu, horizon))
x_init = cp.Parameter(nx)
x_ref = cp.Parameter(nx)

cost = 0
constraints = [x[:, 0] == x_init]
for t in range(horizon):
    cost += (cp.quad_form(x[:, t] - x_ref, Q) + cp.quad_form(u[:, t], R))*(gamma**float(t))

    constraints += [x[:, t+1] == A @ x[:, t] + B @ u[:, t]]
    constraints += [cp.abs(u[:, t]) <= 10]  # Adjusted control input constraints # mai simplificat

# Add terminal cost
cost += (cp.quad_form(x[:, horizon] - x_ref, P))*(gamma**float(horizon))

prob = cp.Problem(cp.Minimize(cost), constraints)

# Serial communication setup
ser = serial.Serial('COM4', 921600)  # Replace 'COM4' with your actual COM port

if not ser.is_open:
    ser.open()

x_ref.value = np.array([0, 0, 0, 0])  # Reference state

# Buffer to store data
data_buffer = []

# Signal handler for graceful termination
def signal_handler(sig, frame):
    print("\nExiting program.")
    # Generate filename with timestamp
    file_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"data_log_{file_timestamp}.txt"
    
    # Save buffer to a file
    with open(filename, "w") as file:
        for entry in data_buffer:
            file.write(entry + "\n")
    ser.close()
    
    print(f"Data saved to {filename}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    print("Reading data from the serial port...")
    while True:
        if ser.in_waiting > 0:
            # Parse the line into a list of floats
            line = ser.readline().decode('utf-8').strip()
            state_data = np.array([float(x) for x in line.split('\t')])
            state_data[0] = state_data[0] * 0.01;
            state_data[1] = state_data[1] * 0.01;
            # Set the initial state
            x_init.value = state_data

            # Solve the MPC problem
            if(np.abs(state_data[2]) < 0.2):
                prob.solve(solver=cp.OSQP)
            
            if prob.status != cp.OPTIMAL:
                #print("Solver failed to find an optimal solution.")
                u_value = 0  # Default value in case of solver failure
            else:
                # Get the optimal control input
                u_value = u.value[0, 0]
                #print(u.value)

            # Print the timestamp, state, and control input on the same line
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds
            state_str = '\t'.join([f"{s:.2f}" for s in state_data])
            print(f"{current_time} - State: {state_str} - Command: {u_value:.2f}")

            # Append the data to buffer
            data_buffer.append(f"{current_time}\t{state_str}\t{u_value:.2f}")

            # Send the control input back to the Arduino
            ser.write(f"{u_value}\n".encode('utf-8'))

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Generate filename with timestamp
    file_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"data_log_{file_timestamp}_rod.txt"
    
    # Save buffer to a file
    with open(filename, "w") as file:
        for entry in data_buffer:
            file.write(entry + "\n")
    
    if ser.is_open:
        ser.close()
    
    print(f"Data saved to {filename}")
