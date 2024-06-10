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

% Asynchronous Unicast PDMM
x_uni_pdmm = x_values;
t_uni_pdmm = 0;
xi_uni_pdmm = zeros(N,N);
c = 0.4;
p = 0.1;

error_uni_pdmm = zeros(max_iter*10,2);
lists_adj = cell(N,1);
for ii=1:N
    lists_adj{ii} = find(A(ii,:)~=0);
end

for ii=1:max_iter*10
    idx = randi(N);

    x_uni_pdmm(idx) = (x_values(idx)+sum(xi_uni_pdmm(idx,lists_adj{idx})))/(1+c*D(idx,idx));
    xi_uni_pdmm_can = (-xi_uni_pdmm(idx,lists_adj{idx})+2*c*x_uni_pdmm(idx))';
    prob = binornd(1, p, size(xi_uni_pdmm_can));
    xi_uni_pdmm(lists_adj{idx},idx) = prob.*xi_uni_pdmm_can+(1-prob).*xi_uni_pdmm(lists_adj{idx},idx);
    t_uni_pdmm = t_uni_pdmm+length(lists_adj{idx});

    error_uni_pdmm(ii,1)=t_uni_pdmm;
    error_uni_pdmm(ii,2)=norm(x_uni_pdmm-x_avg,2);
end

figure
plot(error_uni_pdmm(:,1), error_uni_pdmm(:,2))
set(gca, 'YScale', 'log')


