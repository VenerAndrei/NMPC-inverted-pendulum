import casadi;
import matplotlib.pyplot as plt;
import numpy as np
import time

# Non Linear Model (ODEs)
def dynamic(x_state,u):
    # x = [x dx th dth];
    # System constants
    g = 9.81;
    L = 1;
    l = L / 2;
    m = 0.1;
    M = 1;

    # States
    x = x_state[0]
    dx = x_state[1]
    th = x_state[2];
    dth = x_state[3]

    q_th = 0.001;
    Mf = q_th * dth;

    q_x = 0.01;
    Ff = -q_x * dx;

    f = u[0];
    ddx = (m*g*casadi.sin(th)*casadi.cos(th)  - (7/3)*(f + m*l*dth*dth*casadi.sin(th) + Ff) - (Mf*casadi.cos(th))/l)/(m*casadi.cos(th)*casadi.cos(th) - (7/3)*M);
    ddth = (g*casadi.sin(th) - ddx*casadi.cos(th) - Mf/(m*l))/((7/3)*l);

    return casadi.vertcat(dx,ddx,dth,ddth);

# Numeric Integrator RK4
def rk4(ode, h, x, u):

    k1 = ode( x           , u)
    k2 = ode( x + h/2 * k1, u)
    k3 = ode( x + h/2 * k2, u)
    k4 = ode( x +  h  * k3, u)

    return x + h/6*(k1 + 2*k2 + 2*k3 + k4)

# MPC code
T = 0.2;
N = 100;
u_max = 10;

opti = casadi.Opti();

# The Input
opt_controls = opti.variable(N, 1);
u = opt_controls[:,0];

# States 
opt_states = opti.variable(N+1, 4);
x =  opt_states[:,0];
dx  = opt_states[:,1];
th  = opt_states[:,2];
dth = opt_states[:,3];

# parameters
opt_x0 = opti.parameter(4)
opt_xs = opti.parameter(4)

# init_condition
opti.subject_to(opt_states[0, :] == opt_x0.T)

#Predict the future  
for i in range(N):
    x_next = rk4(dynamic,T,opt_states[i, :].T, opt_controls[i, :].T).T;   # Calculate what the next state will be
    opti.subject_to(opt_states[i+1, :] == x_next)                   # Set new constraint

 
# Define the cost function
# Cost matixes
Q = np.eye(4)
R = np.eye(1)
# cost function
obj = 0  # cost
for i in range(N):
    obj = obj + casadi.mtimes([(opt_states[i, :]-opt_xs.T), Q, (opt_states[i, :]-opt_xs.T).T]) + casadi.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])

opti.minimize(obj)

# boundrary and control conditions
opti.subject_to(opti.bounded(-u_max, u, u_max))

opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0,
                'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}

opti.solver('ipopt', opts_setting)
final_state = np.array([0, 0, 0,  0]) # UPRIGHT position
opti.set_value(opt_xs, final_state)

t0 = 0
init_state = np.array([0 , 0, np.pi/6 , 0]) # DONW POSITION
u0 = np.zeros((N, 1))
current_state = init_state.copy()
next_states = np.zeros((N+1, 4))
x_c = []  # contains for the history of the state
u_c = []
t_c = [t0]  # for the time
xx = []
sim_time = 20.0