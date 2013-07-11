function model = LGMMC_train(K0, bag_idx, bag_prop, para)

% USAGE [y,alpha,y_set,iter,ct,beta,obj_set] = LGMMC_train(K0,X,C,ep,iteration)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Desciption: With inputting the kernel matrix K0, the data X, SVM's
% regularization parameter C, the balance parameter ep and maximum
% iteration number iteration, this function outputs the predicted label vector y, 
% the dual variable alpha in SVM, the generated label set y_set, 
% the number of iteration iter, the cputime ct, 
% the cofficients beta of multiple label-kernel and the objective set obj_set.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
% K0; n*n
% X : d*n
% C : svm parameter

% Output:
% y : n*1, prediction
% alpha: n*1, prediction coefficients
% y_set: T*n 
% beta: 1*T
% obj_set: 1*T

% original implementation based on Yu-Feng Li et al. 
% Tighter and convex maximum margin clustering

bag_struct.bag_idx = bag_idx;
bag_struct.bag_prop = bag_prop;
if ~isfield(para, 'max_iteration')
   para.max_iteration = 10; 
end

if ~isfield(para, 'x')
[U,D] = svd(K0);

total_power = sum(diag(D));
total_power_s = cumsum(diag(D));
power_percent = total_power_s/total_power;

try
    max_idx = find(power_percent<=0.90);
    max_idx = max_idx(end);
catch
    max_idx = find(power_percent<=1-1e-5);
    max_idx = max_idx(end);
end
max_idx = max(max_idx, 50);
idx = 1:max_idx;

U = U(:, idx);
D = D(idx, idx);
para.x = U*D.^0.5;%*U';
end

options.algo='svmclass'; % Choice of algorithm in mklsvm can be either
                         % 'svmclass' or 'svmreg'
%------------------------------------------------------
% choosing the stopping criterion
%------------------------------------------------------
options.stopvariation=0; % use variation of weights for stopping criterion 
options.stopKKT=0;       % set to 1 if you use KKTcondition for stopping criterion    
options.stopdualitygap=1; % set to 1 for using duality gap for stopping criterion

%------------------------------------------------------
% choosing the stopping criterion value
%------------------------------------------------------
options.seuildiffsigma=1e-2;        % stopping criterion for weight variation 
options.seuildiffconstraint=0.01;    % stopping criterion for KKT
options.seuildualitygap=0.01;       % stopping criterion for duality gap

%------------------------------------------------------
% Setting some numerical parameters 
%------------------------------------------------------
options.goldensearch_deltmax=1e-1; % initial precision of golden section search
options.numericalprecision=1e-8;   % numerical precision weights below this value
                                   % are set to zero 
options.lambdareg = 1e-8;          % ridge added to kernel matrix 

%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base 
                                   % variable in the reduced gradient method 
options.nbitermax=3;               % maximal number of iteration  
options.seuil=0;                   % forcing to zero weights lower than this 
options.seuilitermax=2;           % value, for iterations lower than this one 

options.miniter=0;                 % minimal number of iterations 
options.verbosesvm=0;              % verbosity of inner svm algorithm 
options.efficientkernel=1;         % use efficient storage of kernels 
verbose=1;

if ~isfield(para, 'epsilon')
    para.epsilon = 0;
end

% initilize the working set y_set
flag = 1;
[n,~] = size(para.x);
if isfield(para, 'init_y')
    y_set = para.init_y';
else   
    tmpalpha = 1/n*ones(n,1);
    y_set = Max_Violated_y_all_v4(tmpalpha,para.x,bag_struct, para.epsilon);
end

nk = 1;
bestobj = inf;
obj_set = [];

iter = 1;
ttt = cputime;

while flag && iter <= para.max_iteration
     disp([num2str(iter) '....................']);
     % train svm
     [beta,w,b,posw,story,obj] = mklsvm_lgmmc_mod(K0,y_set,ones(n,1), para.C, options, verbose);
     if abs(obj-bestobj) < 0.01*abs(bestobj) % maybe it is a good idea to relax this
            flag = 0;
     end
     if flag         
         bestobj = obj;
         nk = nk + 1;
         options.sigmainit = zeros(1,nk);
         options.sigmainit(1:nk-1) = beta';
         alpha = zeros(n,1);
         alpha(posw) = abs(w);
         
         %% find the most violated label vectore y                 
         y = Max_Violated_y_all_v4(alpha,para.x,bag_struct, para.epsilon);

         %% current objective function:
         %[objective] = compute_obj(K0, w, beta, posw, y_set);
         %fprintf('current objective -theta = %f\n', obj);         
         obj_y = -0.5*alpha'*(K0.* (y'*y))*alpha + sum(alpha);         
         % obj should be smaller than obj_y, when optimal
         violation = obj - obj_y;
         fprintf('violation = %f\n', violation);
         y_set = [y_set;y];
         obj_set = [obj_set,bestobj];     
         iter = iter + 1;
         %% test Max_Violated_y_all_v4
         %y2 = Max_Violated_y_all_v3(alpha,para.x,bag_struct, para.epsilon);
         %obj_y = -0.5*alpha'*(K0.* (y2'*y2))*alpha + sum(alpha);         
         %violation = obj - obj_y;
         %fprintf('violation old = %f\n', violation);
     end 
end
ct = cputime - ttt;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prediction part
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = zeros(n,1);
alpha(posw) = abs(w);
ind = find(beta > eps);
M = zeros(size(K0));
for i = 1:length(ind)
    M = M + beta(ind(i))*y_set(ind(i),:)'*y_set(ind(i),:);
end
[V,D] = eigs(M,1);
predict_model = (alpha'.* V');
y = sign(predict_model*K0);
%y_al = recover_y(K0, beta, y_set);
model.y = y';
%model.y_al = y_al';
model.predict_model = predict_model';
model.M = M;
model.alpha = alpha;
model.y_set = y_set;
model.ct = ct;
model.u = beta;
end

%function [objective, alpha] = compute_obj(K0, w, beta, posw, y_set)
%alpha = zeros(size(K0,1),1);
%alpha(posw) = abs(w);
%ind = find(beta > eps);
%M = zeros(size(K0));
%for i = 1:length(ind)
%    M = M + beta(ind(i))*y_set(ind(i),:)'*y_set(ind(i),:);
%end
%objective = -0.5*alpha' * (K0.* M)*alpha + sum(alpha);
%end