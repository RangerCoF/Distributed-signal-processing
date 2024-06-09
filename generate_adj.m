function A=generate_adj(nodes,r)
% Function used to generate adjacency matrix using a set of "nodes" and the 
% coverage "r"
    [N,~]=size(nodes);
    A = zeros(N,N);
    for ii=1:N
    x_ini = nodes(ii,1);
    y_ini = nodes(ii,2);
        for jj = 1:N
            x_t = nodes(jj,1);
            y_t = nodes(jj,2);
            dist_sq = (x_ini-x_t)^2+(y_ini-y_t)^2;
            if dist_sq <= r^2
                A(ii,jj) = 1;
            end
        end
    end
    A = A-eye(N); % since no self loop, aii is set to 0.

end