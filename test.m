clc; close all; clear all

%% system data
A= [-0.25 0.25; -0.25 -0.25]; B= 0.5 * [1;1];
sys= LTISystem('A', A, 'B', B);

N = 10;
[nx,nu] = size(B);

% define constrainst
sys.x.min = [-10; -10];
sys.x.max = [10; 10];
sys.u.min= [-.1];
sys.u.max= [.1];

% define cost function matrices/quadratic forms
Q = eye(2);
sys.x.penalty = QuadFunction( Q );
R = 0.1; %* eye(2);
sys.u.penalty = QuadFunction( R );

% define terminal cost and set
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
