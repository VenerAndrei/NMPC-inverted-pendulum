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
def rk4(ode, h, xs, u):
    k1 = ode( xs           , u)
    k2 = ode( xs + h/2 * k1, u)
    k3 = ode( xs + h/2 * k2, u)
    k4 = ode( xs +  h  * k3, u)

    return xs + h/6*(k1 + 2*k2 + 2*k3 + k4)


T = 10;
N = 100;
opti = casadi.Opti();
X = opti.variable(4,N+1);
x = X[0,:];
dx = X[1,:];
th = X[2,:];
dth = X[3,:]

U = opti.variable(1,N);
# Define the cost function
# Cost matixes
Q = np.eye(4)
R = np.eye(1)
# cost function
obj = 0  # cost J = x'Qx - u'Ru 
final_state = casadi.vertcat(0,0,0,0);# UPRIGHT position

for i in range(N):
    err = X[:,i] - final_state;
    obj = obj + casadi.mtimes(casadi.mtimes(err.T,Q),err) + U[0,i]*R*U[0,i]

opti.minimize(obj);

dt = T/N;

for k in range(0,N):
    k1 = dynamic( X[:,k]    ,U[0,k])
    k2 = dynamic( X[:,k] + dt/2 * k1, U[0,k])
    k3 = dynamic( X[:,k] + dt/2 * k2, U[0,k])
    k4 = dynamic( X[:,k] +  dt  * k3, U[0,k])
    x_next =  X[:,k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    opti.subject_to(X[:,k+1] == x_next)

opti.subject_to(opti.bounded(-20,U,20));
opti.subject_to(X[:,0] == np.array([0,0,np.pi/6,0]).T);
opti.subject_to(U[0] == 0);

opti.solver('ipopt');

sol = opti.solve();

u_SS = sol.value(U);

x_SS = sol.value(x);
dx_SS = sol.value(dx);

th_SS = sol.value(th);
dth_SS = sol.value(dth);

X_SS = sol.value(X);
# plt.plot(x_SS);
# plt.plot(dx_SS);
# plt.plot(th_SS);
# plt.plot(dth_SS);
plt.plot(X_SS.T)
plt.grid(True)
plt.legend(["x", "dx", "th", "dth"])
plt.title('NMPC Inverted Pendulum - Single Shooting')
plt.show()

