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
    # States
 
    th = x_state[2];
    dth = x_state[3];

    I = (m*L*L)/3;
    H = 1/(m*l*l + I);
    q_th = 0.001;

    ddth = -q_th * H * dth + m*g*l*np.sin(th);

    return np.vstack((x_state[0],x_state[1],dth,ddth));

# Numeric Integrator RK4
def rk4(ode, h, xs, u):
    k1 = ode( xs           , u)
    k2 = ode( xs + h/2 * k1, u)
    k3 = ode( xs + h/2 * k2, u)
    k4 = ode( xs +  h  * k3, u)

    return xs + h/6*(k1 + 2*k2 + 2*k3 + k4)


T = 20; # Seconds


# Variables
N = 1000

x = []     # Cart position
dx = []    # Cart velocity
th = []    # Pendulum angle
dth = []    # Pendulum angular velocity


dt = 0.02;  
step = 0
current_state = [0,0,-3,0];
while(step < N):
    
    current_state = rk4(dynamic,dt,current_state,0);
    #print(current_state)
    th.append(current_state[2]);
    dth.append(current_state[3]); 
    step += 1

timestr = time.strftime("%Y%m%d-%H%M%S")
print(th)
plt.plot(th)
plt.grid(True)
plt.title('NMPC Inverted Pendulum - AngleLog')
plt.savefig("../plots/AngleLog-{}.png".format(timestr))

plt.show()