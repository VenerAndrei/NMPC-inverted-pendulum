import serial
from datetime import datetime
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import signal
import sys

# System dynamics
A = np.array([
    [1, 0.02, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1.006, 0.02004],
    [0, 0, 0.561, 1.006]
])

B = np.array([
    [0.0002],
    [0.02],
    [-0.000572],
    [-0.05725]
])

# Observer dynamics
C = np.eye(4)
D = np.zeros((4, 1))
Q = np.diag([30, 1, 10, 1])
R = np.diag([1])
gamma = 1

# MPC setup
horizon = 50  # prediction horizon
nx = A.shape[0]  # number of states
nu = B.shape[1]  # number of inputs

x = cp.Variable((nx + nu, horizon + 1))
u = cp.Variable((nu, horizon))
x_init = cp.Parameter(nx + nu)
x_ref = cp.Parameter(nx + nu)

Aext = np.block([
    [A, B],
    [np.zeros((nu, nx)), np.eye(nu)]
])

Bext = np.vstack([B, np.zeros((nu, nu))])
Qext = np.diag([30, 1, 10, 1, 0])  # extended Q matrix
Pext = Qext  # terminal cost matrix
cost = 0

constraints = [x[:, 0] == x_init]

for t in range(horizon):
    cost += cp.quad_form(x[:, t] - x_ref, Qext) + cp.quad_form(u[:, t], R) * (gamma**t)
    constraints += [x[:, t + 1] == Aext @ x[:, t] + Bext @ u[:, t]]
    constraints += [cp.abs(u[:, t]) <= 10]  # control input constraints

# Add terminal cost
cost += cp.quad_form(x[:, horizon] - x_ref, Pext) * (gamma**horizon)

prob = cp.Problem(cp.Minimize(cost), constraints)

# Serial communication setup
ser = serial.Serial('COM4', 921600)  # Replace 'COM4' with your actual COM port

if not ser.is_open:
    ser.open()

x_ref.value = np.array([0, 0, 0, 0, 0])  # Reference state
print(x_ref.shape)
# Buffer to store data
data_buffer = []

prev_state_data = np.zeros(nx)
prev_u_value = np.array([0]);

k = 0;
try:
    print("Reading data from the serial port...")
    while True:
        if ser.in_waiting > 0:
            # print(f"k={k}")
            k += 1
            # Parse the line into a list of floats
            line = ser.readline().decode('utf-8').strip()
            # print(line)
            state_data = np.array([float(x) for x in line.split('\t')])
            
            state_data[0] *= 0.01
            state_data[1] *= 0.01
            # Set the initial state
            # print("STATE, PREV_STATE, PREV_U")
            # print(state_data)
            # print(prev_state_data)
            # print(prev_u_value)
            # print("VAL")
            test = np.linalg.pinv(B) @ (state_data - A @ prev_state_data - B @ prev_u_value)
            x_init.value = np.hstack((state_data, test[0]))
            # print(x_init)
            # Solve the MPC problem
            if np.abs(state_data[2]) < 0.2:
                prob.solve(solver=cp.OSQP)

            # print("SOLVED THE PORB")
            if prob.status != cp.OPTIMAL:
                u_value = 0  # Default value in case of solver failure
            else:
                u_value = u.value[0,0]

          
            prev_u_value = np.array([u_value])

            prev_state_data = np.copy(state_data)

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
        print("CLOSED SERIAL CONENCTION")
    
    print(f"Data saved to {filename}")
