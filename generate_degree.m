function D=generate_degree(A)
% Function used to generate a degree matrix using the adjacency matrix "A"
    [N,~]=size(A);
    D = zeros(N,N);
    for ii=1:N
        D(ii,ii) = sum(A(:,ii));
    end

end