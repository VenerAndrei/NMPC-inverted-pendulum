import casadi;
import matplotlib.pyplot as plt;
import numpy as np
import datetime;
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

def rk4(ode, h, x, u):

    k1 = ode( x           , u)
    k2 = ode( x + h/2 * k1, u)
    k3 = ode( x + h/2 * k2, u)
    k4 = ode( x +  h  * k3, u)

    return x + h/6*(k1 + 2*k2 + 2*k3 + k4)

x_init = casadi.vertcat(0,0,casadi.pi/2,0);
x_state = casadi.vertcat(0,0,0,0);
u = [0];
res = []
# parameters
t_0 = 0. # start time
t_end = 15. # end time
h = 0.01 # step size
t_arr = np.arange(t_0, t_end, h) # array of integration times
n_steps = len(t_arr) # number of steps

# states / control inputs
u_stat = np.array([0.])             # static control input
x_0 = np.array([0, 0, np.pi , 0]) # initial state
x = np.zeros([len(x_0), n_steps])   # holds states at different times as columns
x[:,0] = x_0                        # set initial state for t_0

ocp = casadi.Opti()
nx = 4;
nu = 1;
X = ocp.variable(nx,1);
U = ocp.variable(nu,1);

ocp.subject_to(dynamic(X,U) == 0)
# WE HAVE 2 steady state points
# ocp.set_initial(X,[0,0,np.pi-1,1])
ocp.set_initial(X,[0,0,np.pi-1,1]) 

ocp.solver('ipopt');

SS_sol = ocp.solve();
x_SS = SS_sol.value(X);
u_SS = SS_sol.value(U);
print("Steady state solution: \n x={} \n u={}".format(x_SS, u_SS))

