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
error_rng_gossip = zeros(max_iter*20,2);
for ii=1:max_iter*20
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

% Random gossip
p = 0.8;
x_rng_gossip = x_values;
t_rng_gossip = 0;
error_rng_gossip = zeros(max_iter*20,2);
for ii=1:max_iter*20
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
error_rg_08 = error_rng_gossip;


% Random gossip
p = 0.5;
x_rng_gossip = x_values;
t_rng_gossip = 0;
error_rng_gossip = zeros(max_iter*20,2);
for ii=1:max_iter*20
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
error_rg_05 = error_rng_gossip;

% Random gossip
p = 0.1;
x_rng_gossip = x_values;
t_rng_gossip = 0;
error_rng_gossip = zeros(max_iter*20,2);
for ii=1:max_iter*20
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
error_rg_01 = error_rng_gossip;

% Asynchronous Broadcast PDMM
p = 1.0;
x_brd_pdmm = x_values;
x_received = repmat(x_brd_pdmm, [1, N]);
t_brd_pdmm = 0;
xi_brd_pdmm = zeros(N,N);
c = 0.4;

error_brd_pdmm = zeros(max_iter*10,2);
lists_adj = cell(N,1);
for ii=1:N
    lists_adj{ii} = find(A(ii,:)~=0);
end

for ii=1:max_iter*10
    idx = randi(N);

    x_brd_pdmm(idx) = (x_values(idx)+sum(xi_brd_pdmm(idx,lists_adj{idx})))/(1+c*D(idx,idx));
    prob = binornd(1, p, [length(lists_adj{idx}), 1]);
    x_received(lists_adj{idx}, idx) = prob.*x_brd_pdmm(idx) + (1-prob).*x_received(lists_adj{idx}, idx);
    xi_brd_pdmm(lists_adj{idx},idx) = -xi_brd_pdmm(idx,lists_adj{idx})'+2*c*x_received(lists_adj{idx}, idx);
    t_brd_pdmm = t_brd_pdmm+1;

    error_brd_pdmm(ii,1)=t_brd_pdmm;
    error_brd_pdmm(ii,2)=norm(x_brd_pdmm-x_avg,2);
end
error_brd_10 = error_brd_pdmm;

% Asynchronous Broadcast PDMM
p = 0.8;
x_brd_pdmm = x_values;
x_received = repmat(x_brd_pdmm, [1, N]);
t_brd_pdmm = 0;
xi_brd_pdmm = zeros(N,N);
c = 0.4;

error_brd_pdmm = zeros(max_iter*10,2);
lists_adj = cell(N,1);
for ii=1:N
    lists_adj{ii} = find(A(ii,:)~=0);
end

for ii=1:max_iter*10
    idx = randi(N);

    x_brd_pdmm(idx) = (x_values(idx)+sum(xi_brd_pdmm(idx,lists_adj{idx})))/(1+c*D(idx,idx));
    prob = binornd(1, p, [length(lists_adj{idx}), 1]);
    x_received(lists_adj{idx}, idx) = prob.*x_brd_pdmm(idx) + (1-prob).*x_received(lists_adj{idx}, idx);
    xi_brd_pdmm(lists_adj{idx},idx) = -xi_brd_pdmm(idx,lists_adj{idx})'+2*c*x_received(lists_adj{idx}, idx);
    t_brd_pdmm = t_brd_pdmm+1;

    error_brd_pdmm(ii,1)=t_brd_pdmm;
    error_brd_pdmm(ii,2)=norm(x_brd_pdmm-x_avg,2);
end
error_brd_08 = error_brd_pdmm;

% Asynchronous Broadcast PDMM
p = 0.5;
x_brd_pdmm = x_values;
x_received = repmat(x_brd_pdmm, [1, N]);
t_brd_pdmm = 0;
xi_brd_pdmm = zeros(N,N);
c = 0.4;

error_brd_pdmm = zeros(max_iter*10,2);
lists_adj = cell(N,1);
for ii=1:N
    lists_adj{ii} = find(A(ii,:)~=0);
end

for ii=1:max_iter*10
    idx = randi(N);

    x_brd_pdmm(idx) = (x_values(idx)+sum(xi_brd_pdmm(idx,lists_adj{idx})))/(1+c*D(idx,idx));
    prob = binornd(1, p, [length(lists_adj{idx}), 1]);
    x_received(lists_adj{idx}, idx) = prob.*x_brd_pdmm(idx) + (1-prob).*x_received(lists_adj{idx}, idx);
    xi_brd_pdmm(lists_adj{idx},idx) = -xi_brd_pdmm(idx,lists_adj{idx})'+2*c*x_received(lists_adj{idx}, idx);
    t_brd_pdmm = t_brd_pdmm+1;

    error_brd_pdmm(ii,1)=t_brd_pdmm;
    error_brd_pdmm(ii,2)=norm(x_brd_pdmm-x_avg,2);
end
error_brd_05 = error_brd_pdmm;

% Asynchronous Broadcast PDMM
p = 0.1;
x_brd_pdmm = x_values;
x_received = repmat(x_brd_pdmm, [1, N]);
t_brd_pdmm = 0;
xi_brd_pdmm = zeros(N,N);
c = 0.4;

error_brd_pdmm = zeros(max_iter*10,2);
lists_adj = cell(N,1);
for ii=1:N
    lists_adj{ii} = find(A(ii,:)~=0);
end

for ii=1:max_iter*10
    idx = randi(N);

    x_brd_pdmm(idx) = (x_values(idx)+sum(xi_brd_pdmm(idx,lists_adj{idx})))/(1+c*D(idx,idx));
    prob = binornd(1, p, [length(lists_adj{idx}), 1]);
    x_received(lists_adj{idx}, idx) = prob.*x_brd_pdmm(idx) + (1-prob).*x_received(lists_adj{idx}, idx);
    xi_brd_pdmm(lists_adj{idx},idx) = -xi_brd_pdmm(idx,lists_adj{idx})'+2*c*x_received(lists_adj{idx}, idx);
    t_brd_pdmm = t_brd_pdmm+1;

    error_brd_pdmm(ii,1)=t_brd_pdmm;
    error_brd_pdmm(ii,2)=norm(x_brd_pdmm-x_avg,2);
end
error_brd_01 = error_brd_pdmm;

% Asynchronous Unicast PDMM
p = 1;
x_uni_pdmm = x_values;
t_uni_pdmm = 0;
xi_uni_pdmm = zeros(N,N);
c = 0.4;

error_uni_pdmm = zeros(max_iter*20,2);
lists_adj = cell(N,1);
for ii=1:N
    lists_adj{ii} = find(A(ii,:)~=0);
end

for ii=1:max_iter*20
    idx = randi(N);

    x_uni_pdmm(idx) = (x_values(idx)+sum(xi_uni_pdmm(idx,lists_adj{idx})))/(1+c*D(idx,idx));
    xi_uni_pdmm_can = (-xi_uni_pdmm(idx,lists_adj{idx})+2*c*x_uni_pdmm(idx))';
    prob = binornd(1, p, size(xi_uni_pdmm_can));
    xi_uni_pdmm(lists_adj{idx},idx) = prob.*xi_uni_pdmm_can+(1-prob).*xi_uni_pdmm(lists_adj{idx},idx);
    t_uni_pdmm = t_uni_pdmm+length(lists_adj{idx});

    error_uni_pdmm(ii,1)=t_uni_pdmm;
    error_uni_pdmm(ii,2)=norm(x_uni_pdmm-x_avg,2);
end
error_uni_10 = error_uni_pdmm;

% Asynchronous Unicast PDMM
p = 0.8;
x_uni_pdmm = x_values;
t_uni_pdmm = 0;
xi_uni_pdmm = zeros(N,N);
c = 0.4;

error_uni_pdmm = zeros(max_iter*20,2);
lists_adj = cell(N,1);
for ii=1:N
    lists_adj{ii} = find(A(ii,:)~=0);
end

for ii=1:max_iter*20
    idx = randi(N);

    x_uni_pdmm(idx) = (x_values(idx)+sum(xi_uni_pdmm(idx,lists_adj{idx})))/(1+c*D(idx,idx));
    xi_uni_pdmm_can = (-xi_uni_pdmm(idx,lists_adj{idx})+2*c*x_uni_pdmm(idx))';
    prob = binornd(1, p, size(xi_uni_pdmm_can));
    xi_uni_pdmm(lists_adj{idx},idx) = prob.*xi_uni_pdmm_can+(1-prob).*xi_uni_pdmm(lists_adj{idx},idx);
    t_uni_pdmm = t_uni_pdmm+length(lists_adj{idx});

    error_uni_pdmm(ii,1)=t_uni_pdmm;
    error_uni_pdmm(ii,2)=norm(x_uni_pdmm-x_avg,2);
end
error_uni_08 = error_uni_pdmm;

% Asynchronous Unicast PDMM
p = 0.5;
x_uni_pdmm = x_values;
t_uni_pdmm = 0;
xi_uni_pdmm = zeros(N,N);
c = 0.4;

error_uni_pdmm = zeros(max_iter*20,2);
lists_adj = cell(N,1);
for ii=1:N
    lists_adj{ii} = find(A(ii,:)~=0);
end

for ii=1:max_iter*20
    idx = randi(N);

    x_uni_pdmm(idx) = (x_values(idx)+sum(xi_uni_pdmm(idx,lists_adj{idx})))/(1+c*D(idx,idx));
    xi_uni_pdmm_can = (-xi_uni_pdmm(idx,lists_adj{idx})+2*c*x_uni_pdmm(idx))';
    prob = binornd(1, p, size(xi_uni_pdmm_can));
    xi_uni_pdmm(lists_adj{idx},idx) = prob.*xi_uni_pdmm_can+(1-prob).*xi_uni_pdmm(lists_adj{idx},idx);
    t_uni_pdmm = t_uni_pdmm+length(lists_adj{idx});

    error_uni_pdmm(ii,1)=t_uni_pdmm;
    error_uni_pdmm(ii,2)=norm(x_uni_pdmm-x_avg,2);
end
error_uni_05 = error_uni_pdmm;

% Asynchronous Unicast PDMM
p = 0.1;
x_uni_pdmm = x_values;
t_uni_pdmm = 0;
xi_uni_pdmm = zeros(N,N);
c = 0.4;

error_uni_pdmm = zeros(max_iter*20,2);
lists_adj = cell(N,1);
for ii=1:N
    lists_adj{ii} = find(A(ii,:)~=0);
end

for ii=1:max_iter*20
    idx = randi(N);

    x_uni_pdmm(idx) = (x_values(idx)+sum(xi_uni_pdmm(idx,lists_adj{idx})))/(1+c*D(idx,idx));
    xi_uni_pdmm_can = (-xi_uni_pdmm(idx,lists_adj{idx})+2*c*x_uni_pdmm(idx))';
    prob = binornd(1, p, size(xi_uni_pdmm_can));
    xi_uni_pdmm(lists_adj{idx},idx) = prob.*xi_uni_pdmm_can+(1-prob).*xi_uni_pdmm(lists_adj{idx},idx);
    t_uni_pdmm = t_uni_pdmm+length(lists_adj{idx});

    error_uni_pdmm(ii,1)=t_uni_pdmm;
    error_uni_pdmm(ii,2)=norm(x_uni_pdmm-x_avg,2);
end
error_uni_01 = error_uni_pdmm;

figure
hold on
plot(error_rg_10(:,1), error_rg_10(:,2),... 
    'DisplayName', 'Randomized Gossip (p=1.0)')
plot(error_rg_08(:,1), error_rg_08(:,2),...
    'DisplayName', 'Randomized Gossip (p=0.8)')
plot(error_rg_05(:,1), error_rg_05(:,2),...
    'DisplayName', 'Randomized Gossip (p=0.5)')
plot(error_rg_01(:,1), error_rg_01(:,2),...
    'DisplayName', 'Randomized Gossip (p=0.1)')
set(gca, 'YScale', 'log')
xlim([0 1e5])
ylabel('$|\!|\mathbf{x}(k)-x_{avg}\mathbf{1}|\!|_2$', 'interpreter', 'latex')
xlabel("transmissions")
box on
legend



figure
hold on
plot(error_brd_10(:,1), error_brd_10(:,2),...
    'DisplayName', 'PDMM (Broadcast, p=1.0)')
plot(error_brd_08(:,1), error_brd_08(:,2),...
    'DisplayName', 'PDMM (Broadcast, p=0.8)')
plot(error_brd_05(:,1), error_brd_05(:,2),...
    'DisplayName', 'PDMM (Broadcast, p=0.5)')
plot(error_brd_01(:,1), error_brd_01(:,2),...
    'DisplayName', 'PDMM (Broadcast, p=0.1)')
set(gca, 'YScale', 'log')
ylabel('$|\!|\mathbf{x}(k)-x_{avg}\mathbf{1}|\!|_2$', 'interpreter', 'latex')
xlabel("transmissions")
box on
legend

figure
hold on
plot(error_uni_10(:,1), error_uni_10(:,2),...
    'DisplayName', 'PDMM (Unicast, p=1.0)')
plot(error_uni_08(:,1), error_uni_08(:,2),...
    'DisplayName', 'PDMM (Unicast, p=0.8)')
plot(error_uni_05(:,1), error_uni_05(:,2),...
    'DisplayName', 'PDMM (Unicast, p=0.5)')
plot(error_uni_01(:,1), error_uni_01(:,2),...
    'DisplayName', 'PDMM (Unicast, p=0.1)')
set(gca, 'YScale', 'log')
ylabel('$|\!|\mathbf{x}(k)-x_{avg}\mathbf{1}|\!|_2$', 'interpreter', 'latex')
xlabel("transmissions")
box on
legend
xlim([0 2e6])


%{
figure
hold on
plot(error_rg_10(:,1), error_rg_10(:,2),... 
    'DisplayName', 'Randomized Gossip (p=1)', 'LineStyle', '--')
plot(error_rg_08(:,1), error_rg_08(:,2),...
    'DisplayName', 'Randomized Gossip (p=0.8)', 'LineStyle', '--')
plot(error_rg_06(:,1), error_rg_06(:,2),...
    'DisplayName', 'Randomized Gossip (p=0.6)', 'LineStyle', '--')
plot(error_brd_10(:,1), error_brd_10(:,2),...
    'DisplayName', 'PDMM (Broadcast, p=1.0)')
plot(error_brd_08(:,1), error_brd_08(:,2),...
    'DisplayName', 'PDMM (Broadcast, p=0.8)')
plot(error_brd_06(:,1), error_brd_06(:,2),...
    'DisplayName', 'PDMM (Broadcast, p=0.6)')
plot(error_uni_10(:,1), error_uni_10(:,2),...
    'DisplayName', 'PDMM (Unicast, p=1.0)', 'LineStyle', '-.')
plot(error_uni_08(:,1), error_uni_08(:,2),...
    'DisplayName', 'PDMM (Unicast, p=0.8)', 'LineStyle', '-.')
plot(error_uni_06(:,1), error_uni_06(:,2),...
    'DisplayName', 'PDMM (Unicast, p=0.6)', 'LineStyle', '-.')
plot(error_uni_04(:,1), error_uni_04(:,2),...
    'DisplayName', 'PDMM (Unicast, p=0.4)', 'LineStyle', '-.')
xlim([0 2e5])
ylim([1e-8 1e4])
set(gca, 'YScale', 'log')
legend
%}