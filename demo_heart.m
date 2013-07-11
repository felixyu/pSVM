clear all;
init();
load heart.mat;

kernel_type = 'linear';
K = kernel_f(data, kernel_type);
trK = K(split.train_data_idx, split.train_data_idx);
teK = K(split.test_data_idx, split.train_data_idx);

%% kernel InvCal
para.method = 'InvCal';
para.C = 1;
para.ep = 0;
result_invcal = test_all_method(split, trK ,teK, para);

%% kernel alter-pSVM with anealing
para.C = 1; % empirical loss weight
para.C_2 = 1; % proportion term weight
para.ep = 0;
para.method = 'alter-pSVM';
N_random = 20;
result = [];
obj = zeros(N_random,1);
for pp = 1:N_random
    para.init_y = ones(length(trK),1);
    r = randperm(length(trK));
    para.init_y(r(1:floor(length(trK)/2))) = -1;
    result{pp} = test_all_method(split, trK, teK, para);
    obj(pp) = result{pp}.model.obj;
end
[mm,id] = min(obj);
result_alter = result{id};

%% kernel conv-pSVM
% with SVM form as in Li et al. Tigher and convex maximum margin clustering
para.method = 'conv-pSVM';
para.C = 1; % empirical loss weight
result_conv = test_all_method(split, trK, teK, para);

result_invcal
result_alter
result_conv


