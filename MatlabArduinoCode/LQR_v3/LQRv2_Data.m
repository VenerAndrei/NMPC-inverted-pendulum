% Define the filename
filename = 'LQRv2_Data.log';

% Open the file
fileID = fopen(filename, 'r');

% Read the data
data = fscanf(fileID, '%f %f %f %f', [4 Inf]);

% Close the file
fclose(fileID);

% Transpose the data to get columns
data = data';

% Separate the data into individual variables
position = data(:, 1);
velocity = data(:, 2);
pendulum_angle = data(:, 3);
angular_velocity = data(:, 4);

% Define the time array
time_interval = 0.02; % seconds
num_readings = size(data, 1);
time = (0:num_readings-1) * time_interval;

% Plot the data against the timeline
figure;

subplot(4, 1, 1);
plot(time, position, '-o');
title('Position vs. Time');
xlabel('Time (s)');
ylabel('Position (cm)');
yline(0, 'r--'); % Red dotted horizontal line at y=0

subplot(4, 1, 2);
plot(time, velocity, '-o');
title('Velocity vs. Time');
xlabel('Time (s)');
ylabel('Velocity (cm/s)');
yline(0, 'r--'); % Red dotted horizontal line at y=0

subplot(4, 1, 3);
plot(time, pendulum_angle, '-o');
title('Pendulum Angle vs. Time');
xlabel('Time (s)');
ylabel('Pendulum Angle (rad)');
yline(0, 'r--',); % Red dotted horizontal line at y=0

subplot(4, 1, 4);
plot(time, angular_velocity, '-o');
title('Angular Velocity vs. Time');
xlabel('Time (s)');
ylabel('Angular Velocity (rad/s)');
yline(0, 'r--'); % Red dotted horizontal line at y=0

% Display the parsed data (optional)
disp('Position:');
disp(position);
disp('Velocity:');
disp(velocity);
disp('Pendulum Angle:');
disp(pendulum_angle);
disp('Angular Velocity:');
disp(angular_velocity);
