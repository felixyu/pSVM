function [model] = alternating_svm_dual_with_slack(K, Bag_idx, Bag_prop, para, data)

% X N*d, features
% Bag_idx  N*1
% Bag_prop  S*1

if (~isfield(para, 'verbose'))
    para.verbose = 1;
end

if ~isfield(para, 'SVR')
    para.SVR = 0;
end

if ~isfield(para, 'maxiter')
    para.maxiter = 100;
end

model.bag_prop = Bag_prop;
model.bag_idx = Bag_idx;

model.bag_weight = zeros(length(model.bag_prop),1);
for i = 1:length(model.bag_prop)
    model.bag_weight(i) = length(find(model.bag_idx == i));
end

%% initialize
if length(Bag_prop) == length(Bag_idx) % simply initlize as the given labels if only one instance per bag
    model.y = Bag_prop(Bag_idx);
    model.y(model.y==0) = -1;    
elseif isfield(para, 'init_y')
    model.y = para.init_y;
else % random initlization
    model.y = ones(length(Bag_idx),1);
    rand_p = randperm(length(Bag_idx));
    model.y(rand_p(1:floor(length(Bag_idx)/2)))= -1;
end

obj_pre = inf;
ifconverge = 0;
iter = 1;
if para.verbose >= 1
    data_visualization(data, Bag_idx, iter-1, model);
end

while (~ifconverge && iter<=para.maxiter)
    model_pre = model;
    model = optimize_w(K, model, para);
    if isempty(model)
        fprintf('picking up bad parameters...\n');
        model = model_pre; 
        return;
    end
    model = optimize_y(model, para.C_2/para.C, para.ep, para.SVR);
    obj_now = compute_obj(K, model, para);
    eps = abs(obj_now - obj_pre);
    
    if eps< 0.01
        ifconverge = 1;
    else
        fprintf('%d-th iteration ', iter);
        fprintf('obj = %f\n-----\n', obj_now);
        obj_pre = obj_now;
        iter = iter+1;
    end
    if para.verbose >= 1
        data_visualization(data, Bag_idx, iter-1, model);
    end
end
model.iter = iter;
model.obj = obj_now;
end


function [model] = optimize_y(model, C, ep, SVR)

f = model.y_real;
y_n = -ones(size(f,1),1);
y_p = ones(size(f,1),1);

if(SVR)
xi_n = (1 - y_n.*f).^2;
xi_p = (1 - y_p.*f).^2;
else
xi_n = max(1 - y_n.*f, zeros(length(f),1));
xi_p = max(1 - y_p.*f, zeros(length(f),1));
end


xi_flip = xi_n - xi_p;
% the decrease of changing from -1 to +1
% so we want to get the largest

%% now optimize each bag
for idx = 1:length(model.bag_prop)
    current_bag_idx = find(model.bag_idx == idx);
    tau = -length(current_bag_idx):2:length(current_bag_idx); % from all negative to all positive
    xi_flip_current = xi_flip(current_bag_idx);
    
    %% compute the second term of the objective function
    tilda_xi = max(0, model.bag_prop(idx) - ep - tau/2/length(current_bag_idx)-0.5);
    tilda_xi_star = max(0, -model.bag_prop(idx) - ep + tau/2/length(current_bag_idx)+0.5);
    obj_second = C*model.bag_weight(idx)*(tilda_xi + tilda_xi_star);
    
    %% compute the first term of the objective function
    [xi_flip_current_sorted, xi_idx_sorted] = sort(xi_flip_current, 'descend');
    obj_decrease = [0; cumsum(xi_flip_current_sorted)];
    obj_first = sum(xi_n(current_bag_idx)) - obj_decrease;
    obj_proportion = obj_first + obj_second';
    [~,num_to_flip] = min(obj_proportion);
    num_to_flip = num_to_flip-1;
    y_n(current_bag_idx(xi_idx_sorted(1:num_to_flip))) = ones(num_to_flip,1); % flip signs of each bag
    %% record
    model.xi_2(idx) = tilda_xi(num_to_flip+1);
    model.xi_3(idx) = tilda_xi_star(num_to_flip+1);    
end
model.y = y_n;
end

function [ model ] = optimize_w(K, model, para)
    if (para.SVR == 0)
        model_svm = regular_svm_wrapper(K, model.y, para);
    else
        model_svm = regular_svr_wrapper(K, model.y, para);
    end
    if isempty(model_svm)
        model = [];
        return;
    end
    model.support_v = model_svm.support_v;
    model.alp = model_svm.alp; % update the model
    model.b = model_svm.b;
    model.y_real = model_svm.y_real; % predicted y
end

function [objective, obj_1, obj_2] = compute_obj(K, model, para)
    if para.SVR == 0
        xi = max(1 - model.y.*model.y_real, zeros(length(model.y),1));
    else
        xi = 0.5*(1 - model.y.*model.y_real).^2;
    end
    obj_1 = para.C * sum(xi) + 0.5* quad_form(model.alp, K(model.support_v, model.support_v));
    obj_2 = para.C_2*sum((model.xi_2+ model.xi_3)*model.bag_weight);
    objective = obj_1 + obj_2;
end
