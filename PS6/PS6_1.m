%% Code for Problem Set 6, Problem 1 (2), (3)
clear;
clc

%% Initialize the matrix
pf = [0.3, 0.7, 0, 0, 0; 0.3, 0, 0.7, 0, 0; 0.3, 0, 0, 0.7, 0; 0.3, 0, 0, 0, 0.7; 0.3, 0, 0, 0, 0.7];
pb= [0.7, 0.3, 0, 0, 0; 0.7, 0, 0.3, 0, 0; 0.7, 0, 0, 0.3, 0; 0.7, 0, 0, 0, 0.3; 0.7, 0, 0, 0, 0.3];

rf = [3, 0, 0, 0, 0; 3, 0, 0, 0, 0; 3, 0, 0, 0, 0; 3, 0, 0, 0, 0; 3, 0, 0, 0, 10];
rb = rf;

gamma = 0.9;
v = zeros(1, 5);
vstar = zeros(1, 5);
q = zeros(5, 2);

%% Begin Iteration
while sum(vstar - v) == 0 || max(vstar - v) >= 0.001
    v = vstar;
    for i = 1 : 5
        v1 = pf(i, :) .* (rf(i, :) + gamma * v);
        v2 = pb(i, :) .* (rb(i, :) + gamma * v);
        q(i, :) = [sum(v1),  sum(v2)];
        vstar(i) = max(sum(v1), sum(v2));
    end
end

vstar % the optimal v

q % optimal state-value function



