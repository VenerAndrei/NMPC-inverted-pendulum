import casadi;
import matplotlib.pyplot as plt;
import numpy as np
import time
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# Non Linear Model (ODEs)
def dynamic(x_state,u):
    # x = [x dx th dth];
    # System constants
    g = 9.81;
    L = 0.6;
    l = L / 2;
    m = 0.2;
    M = 0.7;

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
    ddth = (M*g*casadi.sin(th) - casadi.cos(th)*(f + m*l*dth*dth*casadi.sin(th) + Ff) - (M*Mf)/(m*l))/((7/3)*M*l - m*l*casadi.cos(th)*casadi.cos(th))

    return casadi.vertcat(dx,ddx,dth,ddth);

# Numeric Integrator RK4
def rk4(ode, h, xs, u):
    k1 = ode( xs           , u)
    k2 = ode( xs + h/2 * k1, u)
    k3 = ode( xs + h/2 * k2, u)
    k4 = ode( xs +  h  * k3, u)

    return xs + h/6*(k1 + 2*k2 + 2*k3 + k4)

opti = casadi.Opti();

T = 20; # Seconds
N = 40; # Prediction Horizon

# Parameters
opt_x0 = opti.parameter(4) # initial state
ref = opti.parameter(4) # reference state

# Variables
X = opti.variable(4,N+1);
x = X[0,:];     # Cart position
dx = X[1,:];    # Cart velocity
th = X[2,:];    # Pendulum angle
dth = X[3,:]    # Pendulum angular velocity

# Control 
U = opti.variable(1,N); # Force

dt = 0.05;  # Discretization time

# Input Constraints
opti.subject_to(opti.bounded(-10,U,10));

# Initial Conditions
opti.subject_to(X[:,0] == opt_x0);
for k in range(0,N):
    k1 = dynamic( X[:,k]    ,U[0,k])
    k2 = dynamic( X[:,k] + dt/2 * k1, U[0,k])
    k3 = dynamic( X[:,k] + dt/2 * k2, U[0,k])
    k4 = dynamic( X[:,k] +  dt  * k3, U[0,k])
    x_next =  X[:,k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    opti.subject_to(X[:,k+1] == x_next)

# Cost matixes
Q = np.eye(4)
R = np.eye(1)

# Cost function
obj = 0  # cost J = x'Qx - u'Ru 
for i in range(N):
    err = X[:,i] - ref;
    obj = obj + casadi.mtimes(casadi.mtimes(err.T,Q),err) + U[0,i]*R*U[0,i]

opti.minimize(obj);
opts_setting = {'ipopt.print_level': 0, 'print_time': 0,'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
opti.solver('ipopt', opts_setting);

final_state = np.array([0,0,0,0]); # UPRIGHT position
init_state = np.array([0,0,np.pi,0]); # Initial position

opti.set_value(ref, final_state);

current_state = init_state.copy();
u0 = np.zeros((1,N));
next_states = np.zeros((4,N+1))

# FOR LOGGING
U_log = np.zeros((1,500));
X_log = np.zeros((4,500));

mpciter = 0;
while(np.linalg.norm(current_state - final_state) > 1e-3 and mpciter < 500):
    
    opti.set_value(opt_x0, current_state); # Set the constraint again
    opti.set_initial(U,u0); # RESET the U variable (INPUTS)
    opti.set_initial(X,next_states) # RESET the X variable(STATES)

    sol = opti.solve();

    u_solved = sol.value(U);
    x_solved = sol.value(X);

    current_state = rk4(dynamic,dt,current_state,u_solved);
    u0 = u_solved[0];

    print(mpciter,np.linalg.norm(current_state - final_state),x_solved[:,0],u_solved[0]);
    U_log[0,mpciter] = u_solved[0];
    X_log[:,mpciter] = x_solved[:,0];
    mpciter = mpciter + 1;

X_log = X_log[:,0:mpciter];
U_log = U_log[0,0:mpciter];

timestr = time.strftime("%Y%m%d-%H%M%S")

plt.plot(X_log.T)
plt.grid(True)
plt.legend(["x", "dx", "th", "dth"])
plt.title('NMPC Inverted Pendulum - Multiple Shooting')
plt.savefig("../plots/CartPoleLog-{}.png".format(timestr))

plt.show()

L = 1;
fig = plt.figure(figsize=(5,5));
ax = fig.add_subplot(autoscale_on=False, xlim=(-3.5,3.5), ylim=(-2,2));
ax.set_aspect('equal')
ax.grid();

cart_line, = ax.plot([], [], 'o-', lw=8);
line, = ax.plot([], [], 'o-', lw=2);

def animate(i):
    cart_pos = X_log[0,i];
    thetha = X_log[2,i];
    x_pend = cart_pos + L*np.sin(thetha);
    y_pend = L*np.cos(thetha);
    line.set_data([cart_pos, x_pend],[0, y_pend]);
    cart_line.set_data([cart_pos - 0.2, cart_pos + 0.2],[0,0])
    return line,cart_line

ani = animation.FuncAnimation(fig, animate, frames=mpciter, blit=True);
ani.save('../gifs/SwingUp-{}.gif'.format(timestr), writer='ffmpeg', fps=15)
plt.show()
