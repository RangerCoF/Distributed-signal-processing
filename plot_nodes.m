function plot_nodes(nodes,edges)
% Function used to plot nodes and edges
    figure;
    scatter(nodes(:,1),nodes(:,2),25,'ob','filled')
    [len1,~] = size(edges);
    hold on
    for ii = 1:len1
        plot([nodes(edges(ii,1),1),nodes(edges(ii,2),1)], ...
            [nodes(edges(ii,1),2),nodes(edges(ii,2),2)],...
            'r','linewidth',0.5);
    end
    hold off
    axis([0 100 0 100]);
end
