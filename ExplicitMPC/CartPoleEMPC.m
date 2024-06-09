clc; close all; clear all

%% system data
% Define constants
M = 0.018;
m = 0.135;
b = 0;
q = 0;
g = 9.8;
l_tot = 0.47;
l = l_tot / 2;
Ts = 0.02;
I = (m * l_tot^2) / 3;

% System matrices
H = 1 / (m * l^2 + I);
A = [0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 (m * g * l) * H 0];
B = [0; 1; 0; -m * l * H];
C = eye(4);
D = zeros(4, 1);

% Create state-space model
sys_ss = ss(A, B, C, D, 'statename', {'x', 'dx', 'th', 'dth'}, 'inputname', {'u'}, 'outputname', {'x', 'dx', 'th', 'dth'});
sys_ss_d = c2d(sys_ss, Ts);

sys= LTISystem('A', sys_ss_d.A, 'B', sys_ss_d.B);

N = 10;
[nx,nu] = size(B);

% define constrainst
sys.x.min = [-15; -15; -0.3; -0.3];
sys.x.max = [15; 15; 0.3; 0.3];
sys.u.min= [-.5];
sys.u.max= [.5];

% define cost function matrices/quadratic forms
Q = diag([50, 1, 10, 1]);
R = 1 * eye(1);

sys.x.penalty = QuadFunction( Q );
sys.u.penalty = QuadFunction( R );

% define terminal cost and set COmentez daca dureaz mult
P = sys.LQRPenalty;
Tset = sys.LQRSet;
sys.x.with('terminalPenalty');
sys.x.with('terminalSet');
sys.x.terminalPenalty = P;
sys.x.terminalSet = Tset;

%% generate the explicit solution

c = MPCController(sys, N);
empc = c.toExplicit();

% get the list of critical regions
CR=empc.partition.Set;
% get the affine laws that characterize each of the CR
for i=1:empc.feedback.Num
    CR(i).addFunction(AffFunction(empc.feedback.Set(i).getFunction('primal').F(1:nu, :),...
                        empc.feedback.Set(i).getFunction('primal').g(1:nu)), 'u0');

    F   = empc.feedback.Set(i).getFunction('primal').F;
    g   = empc.feedback.Set(i).getFunction('primal').g
end

%% plotting

figure; grid on; hold on
title(['critical regions for N=' num2str(N)])
plot(CR)

figure; 
title(['solution for N=' num2str(N)])
CR.fplot('u0')
