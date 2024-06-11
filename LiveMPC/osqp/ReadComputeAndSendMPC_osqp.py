import serial
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix

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

Q = np.diag([100, 1, 100, 1])
R = np.eye(1)

# Horizon
N = 70  # Prediction horizon
nx = A.shape[0]
nu = B.shape[1]

# Construct the extended cost matrices for the QP problem
Q_bar = np.kron(np.eye(N+1), Q)  # Includes an extra block for the initial state
R_bar = np.kron(np.eye(N), R)
H = np.block([
    [Q_bar, np.zeros(((N+1) * nx, N * nu))],
    [np.zeros((N * nu, (N+1) * nx)), R_bar]
])
H_qp = 2 * H  # qpsolvers expects 2*H due to the factor of 1/2 in front of the quadratic term
H_qp_sparse = csc_matrix(H_qp)

# Formulate the equality constraints matrices
Ax = np.kron(np.eye(N+1), -np.eye(nx)) + np.kron(np.eye(N+1, k=-1), A)
Bu = np.kron(np.vstack([np.zeros((1, N)), np.eye(N)]), B)
Aeq = np.hstack([Ax, Bu])
Aeq_sparse = csc_matrix(Aeq)

# Define bounds for inputs (for example -10 to 10)
u_min = -25 * np.ones(N * nu)
u_max = 25 * np.ones(N * nu)

# Define bounds for states (no constraints on states)
x_min = -np.inf * np.ones((N + 1) * nx)
x_max = np.inf * np.ones((N + 1) * nx)

# Combine state and input bounds
bounds_min = np.hstack([x_min, u_min])
bounds_max = np.hstack([x_max, u_max])

# QP vector
f = np.zeros(H_qp_sparse.shape[0])

# Open the serial port
ser = serial.Serial('COM4', 921600)  # Replace 'COM4' with your actual COM port

if not ser.is_open:
    ser.open()

try:
    print("Reading data from the serial port...")
    while True:
        if ser.in_waiting > 0:
            # Parse the line into a list of floats
            line = ser.readline().decode('utf-8').strip()
            state_data = np.array([float(x) for x in line.split('\t')])

            # Update the initial state constraint
            beq = np.zeros((N + 1) * nx)
            beq[:nx] = -state_data

            # Solve the QP problem using qpsolvers
            x_opt = solve_qp(P=H_qp_sparse, q=f, G=None, h=None, A=Aeq_sparse, b=beq, lb=bounds_min, ub=bounds_max, solver='osqp')

            # Extract the optimal control input
            u_opt = x_opt[(N+1)*nx:(N+1)*nx + nu]
            u_value = u_opt[0]  # Assuming a single input

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
