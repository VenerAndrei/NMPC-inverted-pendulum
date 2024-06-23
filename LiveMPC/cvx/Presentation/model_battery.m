import casadi.*;
clear all;close all;
M = 0.018;
m = 0.133;
b = 0;
q = 0;
g = 9.8;

l = 0.485; % 0.52
Ts = 0.05;
I = m*(l^2);

H = 1/(m*l^2 + I);

A = [0      1           0           0;
     0      0           0           0;
     0      0           0           1;
     0      0       (m*g*l)*H       0];
B = [     0;
          1;
          0;
        -m*l*H];


C = eye(4);
D = zeros(4,1);
Q = diag([200,10,100,1]);
R = eye(1);
states = {'x' 'dx' 'th' 'dth'};
inputs = {'u'};
outputs = {'x' 'dx' 'th' 'dth'};

sys_ss = ss(A,B,C,D,'statename',states,'inputname',inputs,'outputname',outputs);
sys_ss_d = c2d(sys_ss,Ts);
K = dlqr(sys_ss_d.A,sys_ss_d.B,Q,R);

simulator.Nsim = 2000;
simulator.x=zeros(4,simulator.Nsim+1);
time = 0:Ts:simulator.Nsim*Ts;
size(time)
u_init=zeros(1,1);
simulator.u=zeros(1,simulator.Nsim);
simulator.x(:,1) = [0;0;0.03;0];
acc = [];
for i=1:simulator.Nsim    
    simulator.u(:,i)=-K*simulator.x(:,i); 
    simulator.x(:,i+1)=sys_ss_d.A*simulator.x(:,i)+sys_ss_d.B*simulator.u(:,i);
end
figure
grid on
subplot(4,1,1)

sgtitle("LQR Cart-Pole")
stairs(time,simulator.x(1,:),'LineWidth',1)
title("x")
xlabel("Time (s)")
ylabel("Position (m)")

subplot(4,1,2)
stairs(time,simulator.x(2,:),'LineWidth',1)
title("dx")
xlabel("Time (s)")
ylabel("Velocity (m/s)")

subplot(4,1,3)
stairs(time,simulator.x(3,:),'LineWidth',1)
title("theta")
xlabel("Time(s) ")
ylabel("Angle (rad)")

subplot(4,1,4)
stairs(time,simulator.x(4,:),'LineWidth',1)
title("dtheta")
xlabel("Time (s)")
ylabel("Angular Velocity (rad/s)")

K
sys_ss_d