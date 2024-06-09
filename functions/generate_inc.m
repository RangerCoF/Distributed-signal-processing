function B=generate_inc(A,edges)
% Function used to generate incidence matrix using adjacency matrix "A"
% and edges
    [N,~]=size(A);
    B=zeros(N,sum(sum(A))/2);
    for ii=1:sum(sum(A))/2
        B(edges(ii,1),ii)=1;
        B(edges(ii,2),ii)=-1;
    end

end
