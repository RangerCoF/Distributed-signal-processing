clc;
clear all;
close all;
% initialization
rng(42)


N = 100; % number of nodes
% randomly distributed nodes
r = 100*sqrt(2*log(N)/N); % coverage of nodes
nodes_xaxis = 100*rand(N,1);
nodes_yaxis = 100*rand(N,1);

% grid of nodes
% r = 10
% nodes_xaxis = [5:10:100,5:10:100,5:10:100,5:10:100,5:10:100,5:10:100,5:10:100,5:10:100,5:10:100,5:10:100]'
% nodes_yaxis = [5*ones(1,10),15*ones(1,10),25*ones(1,10),35*ones(1,10),45*ones(1,10),55*ones(1,10),65*ones(1,10),75*ones(1,10),85*ones(1,10),95*ones(1,10)]'

nodes = [nodes_xaxis,nodes_yaxis]; 
max_iter = 5000;
%xlim([0 100])
%ylim([0 100])

A = generate_adj(nodes,r); % adjacency matrix
D = generate_degree(A); % degree matrix
L = D-A; % Laplacian matrix
edges = generate_edges(A); % edges of the graph
B = generate_inc(A,edges); % incidence matrix
disp("check if L = B*B': "+isequal(L,B*B'))

x_values = 10*randn(N,1)+5;
% x_values = 10*rand(N,1)+5;
x_avg = mean(x_values)*ones(N,1);

% Random gossip
p = 1.0;
x_rng_gossip = x_values;
t_rng_gossip = 0;
error_rng_gossip = zeros(max_iter*5,2);
for ii=1:max_iter*5
    idx1 = randi(N);
    list_adj = [find(A(idx1,:)~=0)];
    idx2 = list_adj(randi(length(list_adj)));
    x_avg_2n = mean(x_rng_gossip([idx1, idx2]));
    prob = binornd(1, p, [2, 1]);
    x_rng_gossip([idx1, idx2]) = prob*x_avg_2n+(1-prob).*x_rng_gossip([idx1, idx2]);
    t_rng_gossip = t_rng_gossip + 2;
    error_rng_gossip(ii,1) = t_rng_gossip;
    error_rng_gossip(ii,2) = norm(x_rng_gossip-x_avg,2);
end
error_rg_10 = error_rng_gossip;


figure
plot(error_rng_gossip(:,1), error_rng_gossip(:,2))
set(gca, 'YScale', 'log')
%}