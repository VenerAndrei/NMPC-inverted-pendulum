% MATLAB script to read data from file and plot states and command

% Load data
data = readtable('data_log_20240611_215407.txt', 'Delimiter', '\t', 'Format', '%s%f%f%f%f%f');

% Extract columns
timestamps = data{:, 1};
x = data{:, 2};
dx = data{:, 3};
theta = data{:, 4};
dtheta = data{:, 5};
command = data{:, 6};

% Convert timestamps to datetime array
timestamps = datetime(timestamps, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');

% Plot the states
figure;
subplot(4, 1, 1);
plot(timestamps, x);
ylabel('x (m)');
title('States vs. Time');
grid on;

subplot(4, 1, 2);
plot(timestamps, dx);
ylabel('dx (m/s)');
grid on;

subplot(4, 1, 3);
plot(timestamps, theta);
ylabel('\theta (rad)');
grid on;

subplot(4, 1, 4);
plot(timestamps, dtheta);
ylabel('d\theta (rad/s)');
xlabel('Time');
grid on;

% Plot the command
figure;
plot(timestamps, command);
title('Command vs. Time');
xlabel('Time');
ylabel('Command (N)');
grid on;
