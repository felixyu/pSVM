function data_visualization(data, bag_idx, iter, model)
% visualize the first 5 bags
    bag_num = min(5, length(unique(bag_idx)));
    for i = 1:bag_num     
        pidx = find(bag_idx==i & model.y==1);
        nidx = find(bag_idx==i & model.y==-1);
        subplot(5, bag_num, iter*bag_num+i); hold(gca,'on');
        title(sprintf('iter=%d, bag idx = %d', iter, i));

        if isfield(model, 'alp')
            w = model.alp'*data(model.support_v,:);
            b = model.b;
            x = min(data(:,1)):0.01:max(data(:,1));
            plot(data(pidx,1),data(pidx,2),'.r',data(nidx,1),data(nidx,2),'.b', x, (-b-w(1)*x)/w(2), '--black');
        else
            plot(data(pidx,1),data(pidx,2),'.r',data(nidx,1),data(nidx,2),'.b');   
        end
        xlim([-1 1]);
        ylim([-1 1]);
    end
end