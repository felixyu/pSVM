function model = invcal_dual(K, Bag_idx, Bag_prop, para)

% X N*d
% Bag_idx  N*1 
% Bag_prop  S*1
% K is a N*N matrix of the training instances

% 12-18 initial implementation
C = para.C;

n = length(Bag_prop);
% to avoid numerical issues
Bag_prop(Bag_prop<1/size(K,1)) = 1/size(K,1);
Bag_prop(Bag_prop>1-1/size(K,1)) = 1-1/size(K,1);

%% reverse calibration, and compute mean for each group
Y = zeros(n,1);
S = zeros(n,1);
ep = zeros(n,1);
for i = 1:n
    p = Bag_prop(i);
    Y(i) = -log(1/p-1);
    ep(i) = para.ep/(p*(1-p));
    S(i) = length(find(Bag_idx == i));
end
k1 = ep - Y;
k2 = ep + Y;
%k1(isnan(k1)) = 0;
%k2(isnan(k2)) = 0;


K_bag = zeros(n,n); % get the kernel matrix on the bag level
for i = 1:n
   for j = 1:i
       K_tmp = K(Bag_idx == i, Bag_idx == j);
       K_bag(i,j) = sum(K_tmp(:));
       K_bag(i,j) = K_bag(i,j)/S(i)/S(j);
       K_bag(j,i) = K_bag(i,j);
   end
end
%K_bag(isnan(K_bag)) = 0;


cvx_begin
    variables alp(n) alp_2(n)
    minimize 0.5 *  quad_form(alp-alp_2, K_bag) + sum(alp.*k1 + alp_2.*k2)
    subject to
        sum(alp-alp_2) == 0;
        alp >= 0;
        alp_2 >= 0;
        alp <= C;
        alp_2 <= C;
cvx_end

model.alp = alp;
model.alp_2 = alp_2;
model.dual = 1;
model.bag_idx = Bag_idx;

%% compute b
% select i s.t. 0<alp_i<C
eps = 1e-20;
idx = find(model.alp > eps & model.alp < C-eps);
b_1 = Y(idx) - ep(idx) - K_bag(idx,:)*(alp - alp_2);

idx = find(model.alp_2 > eps & model.alp_2 < C-eps);
b_2 = Y(idx) + ep(idx) - K_bag(idx,:)*(alp - alp_2);
model.b = mean([b_1; b_2]);
if isnan(model.b)
    model.b = 0;
end

%model.support_v = find(abs(model.alp_2 - model.alp) > eps);
% in regression, seems there is no such concept as support vectors
% ind(abs(model.alp_2 - model.alp) > eps) contains all the alps (almost). so all of
% them needs to be considered in testing

%% get predicted y on the training data
K_test_bag = zeros(size(K,1), size(model.alp,1));
for j = 1:size(model.alp,1)
    K_test_bag(:,j) = mean(K(:, Bag_idx == j), 2);
end
predict_response = K_test_bag * (model.alp - model.alp_2) + model.b;
y = predict_response; y(y>0) = 1; y(y<0) = -1;
model.y = y;
model.y_real = predict_response;

end