import casadi.*;
clear all; close all; clc;

%% Parameters
M = 0.018;
m = 0.135;
b = 0;
q = 0;
g = 9.8;
l_tot = 0.47;
l = l_tot / 2;
Ts = 0.02;
I = (m * l_tot^2) / 3;

%% THEORETICAL MODEL
H = 1 / (m * l^2 + I);

A = [0,   1,         0,          0;
     0,   0,         0,          0;
     0,   0,         0,          1;
     0,   0,   (m*g*l)*H,        0];

B = [0;
     1;
     0;
    -m*l*H];

C = eye(4);
D = zeros(4,1);

Q = diag([100, 5, 100, 1]);
R = 1;

states = {'x', 'dx', 'th', 'dth'};
inputs = {'u'};
outputs = {'x', 'dx', 'th', 'dth'};

sys_ss = ss(A, B, C, D, 'statename', states, 'inputname', inputs, 'outputname', outputs);
sys_ss_d = c2d(sys_ss, Ts);
K = dlqr(sys_ss_d.A, sys_ss_d.B, Q, R);

%% IDENTIFIED MODEL (DMDc)
A_dmdc = [ 1.00318215,  0.0402992084,  0.0142986695, -0.000798676921;
           0.0737403619,  1.00545085,  0.331453472,  -0.0181068159;
          -0.0593877574, -0.0579199012, 0.728317627,  0.0194135524;
          -1.22084510,  -1.25948796,  -5.58794108,   0.567101511 ];

B_dmdc = [ 3.83351103e-04;
           8.18738929e-03;
          -5.02789525e-05;
          -2.69835423e-03 ];

sys_dmdc = ss(A_dmdc, B_dmdc, C, D, Ts);
K_dmdc = dlqr(A_dmdc, B_dmdc, Q, R);

%% Simulation Parameters
Nsim = 200;
time = 0:Ts:Nsim*Ts;
x0 = [0; 0; 0.03; 0];  % initial condition

%% Simulate Theoretical Model
simulator.x = zeros(4, Nsim+1);
simulator.u = zeros(1, Nsim);
simulator.x(:,1) = x0;

for i = 1:Nsim
    simulator.u(:,i) = -K * simulator.x(:,i);
    simulator.x(:,i+1) = sys_ss_d.A * simulator.x(:,i) + sys_ss_d.B * simulator.u(:,i);
end

%% Simulate DMDc Model
sim_dmdc.x = zeros(4, Nsim+1);
sim_dmdc.u = zeros(1, Nsim);
sim_dmdc.x(:,1) = x0;

for i = 1:Nsim
    sim_dmdc.u(:,i) = -K_dmdc * sim_dmdc.x(:,i);
    sim_dmdc.x(:,i+1) = A_dmdc * sim_dmdc.x(:,i) + B_dmdc * sim_dmdc.u(:,i);
end

%% Plot Comparison
figure;
sgtitle("LQR: Theoretical vs DMDc Identified Model");

labels = {'x (m)', 'dx (m/s)', 'theta (rad)', 'dtheta (rad/s)'};
for i = 1:4
    subplot(4,1,i)
    plot(time, simulator.x(i,:), 'b-', 'LineWidth', 1.5); hold on;
    plot(time, sim_dmdc.x(i,:), 'r--', 'LineWidth', 1.5);
    ylabel(labels{i});
    grid on;
    if i == 1
        legend('Theoretical', 'DMDc Identified');
    end
end
xlabel('Time (s)');

%% Plot Control Inputs
figure;
plot(time(1:end-1), simulator.u, 'b', 'LineWidth', 1.5); hold on;
plot(time(1:end-1), sim_dmdc.u, 'r--', 'LineWidth', 1.5);
legend('Theoretical LQR u', 'DMDc LQR u');
xlabel('Time (s)');
ylabel('Control Input');
grid on;
title('Control Effort Comparison');

%% Show Gains
disp('LQR Gain (Theoretical):');
disp(K);

disp('LQR Gain (DMDc):');
disp(K_dmdc);
%% Define and discretize models
ssc = ss(A, B, eye(4), 0);         % Physics-based model
ssc = c2d(ssc, 0.02);              % Discretize with 20 ms sample time

ssd = ss(A_dmdc, B_dmdc, eye(4), 0);  % DMDc (closed-loop identified model)

%% Reconstruct Open-loop system using known controller gain
K = [-12.1732, -11.3985, -59.1152, -13.8610];  % 1x4 row vector
A_open = A_dmdc - B_dmdc * K;                  % Recover open-loop A
ss_open = ss(A_open, B_dmdc, eye(4), 0);       % Build open-loop SS system

%% Extract poles for plotting
p_ssc    = pole(ssc);
p_ssd    = pole(ssd);
p_open   = pole(ss_open);

z_ssc    = tzero(ssc);
z_ssd    = tzero(ssd);
z_open   = tzero(ss_open);

%% Plot pole-zero map with custom markers
figure;
hold on;

% Plot using pzmap (for background axes)
pzmap(ssc);    % blue
pzmap(ssd);    % red
pzmap(ss_open) % green

% Overplot with styled markers
plot(real(p_ssc),  imag(p_ssc),  'bx', 'MarkerSize', 10, 'LineWidth', 2); % Physics poles
plot(real(p_ssd),  imag(p_ssd),  'rx', 'MarkerSize', 10, 'LineWidth', 2); % DMDc poles
plot(real(p_open), imag(p_open), 'gx', 'MarkerSize', 10, 'LineWidth', 2); % Open-loop recovered poles

plot(real(z_ssc),  imag(z_ssc),  'bo', 'MarkerSize', 10, 'LineWidth', 2); % Physics zeros
plot(real(z_ssd),  imag(z_ssd),  'ro', 'MarkerSize', 10, 'LineWidth', 2); % DMDc zeros
plot(real(z_open), imag(z_open), 'go', 'MarkerSize', 10, 'LineWidth', 2); % Open-loop recovered zeros

% Formatting
legend({'Physics poles', 'DMDc poles', 'Recovered Open-loop poles', ...
        'Physics zeros', 'DMDc zeros', 'Recovered Open-loop zeros'}, ...
        'Location', 'best');
grid on;
axis equal;
xlabel('Real');
ylabel('Imaginary');
title('Pole-Zero Map: Physics vs DMDc vs Recovered Open-loop');

