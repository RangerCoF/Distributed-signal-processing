function edges=generate_edges(A)
% Function used to generate edges of the graph using the adjacency matrix
% "A"
    [edges_x,edges_y]=find(A~=0);
    edges = [edges_x,edges_y];
    for ii=1:sum(sum(A))
        if edges(ii,1)>edges(ii,2)
            temp = edges(ii,1);
            edges(ii,1) = edges(ii,2);
            edges(ii,2) = temp;
        end
    end
    edges = unique(edges,'rows');
end