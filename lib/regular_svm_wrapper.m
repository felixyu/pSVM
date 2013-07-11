function [model] = regular_svm_wrapper( K, label, para )
% a regular svm wrapper using libsvm
% using binary labels -1/+1

model_svm = svmtrain(label, [(1:length(K))', K], sprintf('-t 4 -c %f -q', para.C));
model.support_v = full(model_svm.SVs);

if isempty(model.support_v)
    model = [];
    return;
end

model.alp = model_svm.Label(1) * model_svm.sv_coef; % with label
model.b = -model_svm.Label(1) * model_svm.rho;
model.y_real = K(:, model.support_v) * model.alp + model.b;

% alp of libsvm has already taken ground-truth labels into consideration
% if not, we should use:
% model.y_real =  K(:, model.support_v) * (label(model.support_v).*model.alp) + model.b;

% y = model.y_real;
% y(y>0) = 1;
% y(y<0) = -1;

% fprintf('accuracy = %f\n', length(find(y == label))/length(y));
% alp of libsvm has already taken ground-truth labels into consideration
% [a,b,c] = svmpredict(label, [(1:length(K))', K], model_svm);
% model.y_real = c;
end