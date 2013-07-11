function [y, y2] = Max_Violated_y_all_v4(alpha, X, bag_struct, epsilon)

    bag_idx  = bag_struct.bag_idx;
    p = bag_struct.bag_prop;
    
    d = size(X,2);
    n = size(X,1);
    Y_all_pos = zeros(n,d);
    Y_all_neg = zeros(n,d);
    pos_obj = zeros(d,1);
    neg_obj = zeros(d,1);
    
    for i = 1:length(p)  %for each bag
        idx = find(bag_idx == i);
        [y_pos, y_neg, pos_obj_bag, neg_obj_bag] = Max_Violated_y_bag(alpha(idx), X(idx,:), p(i), epsilon);
        pos_obj = pos_obj + pos_obj_bag;
        neg_obj = neg_obj + neg_obj_bag;
        Y_all_pos(idx,:) = y_pos;
        Y_all_neg(idx,:) = y_neg;
    end
    [Y_pos, Y_pos_d] = max(pos_obj);
    [Y_neg, Y_neg_d] = max(neg_obj);
    
    if Y_pos > Y_neg
        y = Y_all_pos(:,Y_pos_d);
        y2 = Y_all_neg(:,Y_neg_d);
    else
        y = Y_all_neg(:,Y_neg_d);
        y2 = Y_all_pos(:,Y_pos_d);
    end
    y = y';
    y2 = y2';

end


% for each bag
function [y_pos, y_neg, pos_obj, neg_obj] = Max_Violated_y_bag(alpha, X, p, epsilon)

% number of positive can range from p+epsilon to p-epsilon
% epsilon can only work when bag size is large
% generally we can set epsilon = 0

[n,d] = size(X);
max_num_of_positive = min(floor(n*(p+epsilon)), n);
min_num_of_positive = max(0, ceil(n*(p-epsilon)));

y_pos = zeros(n,d);
pos_obj = zeros(d,1);
y_neg = zeros(n,d);
neg_obj = zeros(d,1);

%% sub problem 1
for i = 1:d % outer problem 
    t = alpha.* X(:,i);
    % now solve the problem max \sum(t_i * y_i)
    [pos_obj(i), y_pos(:,i)] = lib_compute_flip(t, min_num_of_positive, max_num_of_positive);
    % now solve the problem max \sum(-t_i * y_i)
    [neg_obj(i), y_neg(:,i)] = lib_compute_flip(-t, min_num_of_positive, max_num_of_positive);
end

end

%% for each bag and each dimension
function [obj, y] = lib_compute_flip(t, min_po, max_po)
% y_i are initlized as -1
% we must have min_po <= #positive <= max_po
% max obj
% obj = sum {t_i * y_i}
y = -ones(length(t),1);
[~, idx_sorted] = sort(t, 'descend');
[~, idx_zero] =  min(abs(t));  % this can be speeded up by bisection method!
if idx_zero < min_po
    y(idx_sorted(1:min_po)) = 1;
elseif idx_zero > max_po
    y(idx_sorted(1:max_po)) = 1;
else
    y(idx_sorted(1:idx_zero)) = 1;
end
obj = y'*t;
end
