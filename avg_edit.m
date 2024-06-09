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

nodes = [nodes_xaxis,nodes_yaxis]; 
x_values = 10*randn(N,1)+5;
max_iter = 50000;
xlim([0 100])
ylim([0 100])

A = generate_adj(nodes,r); % adjacency matrix
D = generate_degree(A); % degree matrix
L = D-A; % Laplacian matrix
edges = generate_edges(A); % edges of the graph
B = generate_inc(A,edges); % incidence matrix
disp("check if L = B*B': "+isequal(L,B*B'));

nodes2 = nodes(1:end-1,:);
nodes3 = [nodes; 100*rand(1,2)];

A2 = generate_adj(nodes2,r); % adjacency matrix
D2 = generate_degree(A2); % degree matrix
L2 = D2-A2; % Laplacian matrix
edges2 = generate_edges(A2); % edges of the graph
B2 = generate_inc(A2,edges2); % incidence matrix

A3 = generate_adj(nodes3,r); % adjacency matrix
D3 = generate_degree(A3); % degree matrix
L3 = D3-A3; % Laplacian matrix
edges3 = generate_edges(A3); % edges of the graph
B3 = generate_inc(A3,edges3); % incidence matrix


x_values2 = x_values(1:N-1);
x_values3 = [x_values; 10*randn(1,1)+5];
x_avg = mean(x_values)*ones(N,1);
x_avg2 = mean(x_values2)*ones(N-1,1);
x_avg3 = mean(x_values3)*ones(N+1,1);

x_rng = x_values;
t_rng = 0;
max_iter = 10000;
error_rng = zeros(max_iter,2);
for ii=1:max_iter
    idx1 = randi(N);
    list_adj = [find(A(idx1,:)~=0)];
    idx2 = list_adj(randi(length(list_adj)));
    e_i = zeros(N,1);
    e_j = zeros(N,1);
    e_i(idx1) = 1;
    e_j(idx2) = 1;
    W_ij = eye(N)-0.5*(e_i-e_j)*(e_i-e_j)';
    x_rng = W_ij*x_rng;
    t_rng = t_rng + 2;
    error_rng(ii,1) = t_rng;
    error_rng(ii,2) = norm(x_rng-x_avg,2);
end

semilogy(error_rng(:,1),error_rng(:,2))

x_rng2 = x_rng(1:N-1);
t_rng2 = t_rng;
max_iter2 = 30000;
error_rng2 = zeros(max_iter2,2);
for ii=1:max_iter2
    idx1 = randi(N-1);
    list_adj = [find(A2(idx1,:)~=0)];
    idx2 = list_adj(randi(length(list_adj)));
    e_i = zeros(N-1,1);
    e_j = zeros(N-1,1);
    e_i(idx1) = 1;
    e_j(idx2) = 1;
    W_ij = eye(N-1)-0.5*(e_i-e_j)*(e_i-e_j)';
    x_rng2 = W_ij*x_rng2;
    t_rng2 = t_rng2 + 2;
    error_rng2(ii,1) = t_rng2;
    error_rng2(ii,2) = norm(x_rng2-x_avg2,2);
end

x_rng3 = [x_rng; x_values3(end)];
t_rng3 = t_rng;
max_iter3 = 30000;
error_rng3 = zeros(max_iter3,2);
for ii=1:max_iter3
    idx1 = randi(N+1);
    list_adj = [find(A3(idx1,:)~=0)];
    idx2 = list_adj(randi(length(list_adj)));
    e_i = zeros(N+1,1);
    e_j = zeros(N+1,1);
    e_i(idx1) = 1;
    e_j(idx2) = 1;
    W_ij = eye(N+1)-0.5*(e_i-e_j)*(e_i-e_j)';
    x_rng3 = W_ij*x_rng3;
    t_rng3 = t_rng3 + 2;
    error_rng3(ii,1) = t_rng3;
    error_rng3(ii,2) = norm(x_rng3-x_avg3,2);
end

error_rmv = [error_rng; error_rng2];
error_add = [error_rng; error_rng3];

figure
plot(error_rmv(:,1),error_rmv(:,2))
set(gca, 'YScale', 'log')
xlabel('transmissions')
ylabel('error')
%semilogy(error_rmv(:,2))
figure
plot(error_add(:,1),error_add(:,2))
set(gca, 'YScale', 'log')
xlabel('transmissions')
ylabel('error')
%semilogy(error_add(:,2))