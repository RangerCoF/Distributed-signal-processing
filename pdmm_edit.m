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

c = 0.4;

% 1
max_iter = 10000;
x_brd_pdmm = x_values;
t_brd_pdmm = 0;
xi_brd_pdmm = zeros(N,N);

error_brd_pdmm = zeros(max_iter*10,2);
lists_adj = cell(N,1);
for ii=1:N
    lists_adj{ii} = find(A(ii,:)~=0);
end

for ii=1:max_iter*10
    idx = randi(N);

    x_brd_pdmm(idx) = (x_values(idx)+sum(xi_brd_pdmm(idx,lists_adj{idx})))/(1+c*D(idx,idx));
    xi_brd_pdmm(lists_adj{idx},idx)=-xi_brd_pdmm(idx,lists_adj{idx})+2*c*x_brd_pdmm(idx);
    t_brd_pdmm = t_brd_pdmm+1;

    error_brd_pdmm(ii,1)=t_brd_pdmm;
    error_brd_pdmm(ii,2)=norm(x_brd_pdmm-x_avg,2);
end

%2
max_iter2 = 10000;
x_brd_pdmm2 = x_values(1:N-1);
t_brd_pdmm2 = t_brd_pdmm;
xi_brd_pdmm2 = xi_brd_pdmm(1:N-1,1:N-1);

error_brd_pdmm2 = zeros(max_iter2*10,2);
lists_adj = cell(N-1,1);
for ii=1:N-1
    lists_adj{ii} = find(A2(ii,:)~=0);
end

for ii=1:max_iter2*10
    idx = randi(N-1);

    x_brd_pdmm2(idx) = (x_values2(idx)+sum(xi_brd_pdmm2(idx,lists_adj{idx})))/(1+c*D2(idx,idx));
    xi_brd_pdmm2(lists_adj{idx},idx)=-xi_brd_pdmm2(idx,lists_adj{idx})+2*c*x_brd_pdmm2(idx);
    t_brd_pdmm2 = t_brd_pdmm2+1;

    error_brd_pdmm2(ii,1)=t_brd_pdmm2;
    error_brd_pdmm2(ii,2)=norm(x_brd_pdmm2-x_avg2,2);
end

%3
max_iter3 = 10000;
x_brd_pdmm3 = x_values3;
x_brd_pdmm3(1:N) = x_values;
t_brd_pdmm3 = t_brd_pdmm;
xi_brd_pdmm3 = zeros(N+1,N+1);
xi_brd_pdmm3(1:N,1:N) = xi_brd_pdmm;

error_brd_pdmm3 = zeros(max_iter3*10,2);
lists_adj = cell(N+1,1);
for ii=1:N+1
    lists_adj{ii} = find(A3(ii,:)~=0);
end

for ii=1:max_iter3*10
    idx = randi(N+1);

    x_brd_pdmm3(idx) = (x_values3(idx)+sum(xi_brd_pdmm3(idx,lists_adj{idx})))/(1+c*D3(idx,idx));
    xi_brd_pdmm3(lists_adj{idx},idx)=-xi_brd_pdmm3(idx,lists_adj{idx})+2*c*x_brd_pdmm3(idx);
    t_brd_pdmm3 = t_brd_pdmm3+1;

    error_brd_pdmm3(ii,1)=t_brd_pdmm3;
    error_brd_pdmm3(ii,2)=norm(x_brd_pdmm3-x_avg3,2);
end



error_rmv = [error_brd_pdmm; error_brd_pdmm2];
error_add = [error_brd_pdmm; error_brd_pdmm3];

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
%}