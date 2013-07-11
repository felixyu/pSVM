function [xsup,w,b,pos,timeps,alpha,obj]=svmclass_lgmmc_mod(K, C)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% new optimization problem
%   min 1/2 x'Hx - e'x
%   s.t. 0 <= x <= c

% alpha and w are the same
% b = 0

% Input:
%   K: n*n hesian matrix
%   e: n*1 linear_term
%   c: upper bound of box constraint
%   x0: initialization

% OUTPUT
%
% xsup	coordinates of the Support Vector
% w      weight
% b		bias
% pos    position of Support Vector
% timeps time for processing the scalar product
% alpha  Lagragian multiplier
% obj    Value of Objective function
%

n = size(K,2);
for i = 1:n
    K(i,i) = K(i,i) + 1e-9; 
end

tic
para.C = C;
%model = simplemkl_dual_libvm(K, para); % yu-feng li's approach
%model = simplemkl_dual_old(K, para); % yu-feng li's approach, with gurobi
model = simplemkl_dual(K, para); % new approach

ind = full(model.SVs);
x = zeros(n,1);
x(ind) = model.sv_coef;
%fval = 0.5*x(ind)'*(K(ind,ind) + 1/C * eye(length(ind)))*x(ind); % yu-feng's objective function
%fval = 0.5*x(ind)'*K(ind,ind)*x(ind); % felix's objective function
fval =  0.5*quad_form(x,K) - sum(x);
timeps = toc;

%% assignment
xsup  = [];
pos = find(abs(x) > eps);
w = x(pos);
b = 0;
alpha = w;
obj = -fval;
end


function model = simplemkl_dual_libvm(K, para)
% optimization problem
%   min 1/2 x'Hx + e'x
%   s.t. 0 <= x <= c
%            x'1 = 1
% H = K

% Felix: it is actually solving
%   min  x'Hx
%   s.t. 0 <= x <= 1
%        x'1 = n/2

n = size(K,1);
C = para.C;
K = K + 1/C * eye(n);
opt = ['-q -s 2 -t 4 -c ' num2str(n/2)];
K1 = [(1:n)',K];
%%% for standard libsvm
model = svmtrain(ones(n,1),K1,opt);
end

function model = simplemkl_dual_old(K, para)
% Felix: what they are actually solving is, this is my guess
%   min  x'Hx
%   s.t. 0 <= x <= 1
%        x'1 = n/2

% K = X*X' N*N
% w d*1
n = size(K,1);
Y = ones(n,1);
C = para.C;
K = K + 1/C * eye(n);
cvx_solver Gurobi % in order to run on the cluster
cvx_quiet true
cvx_begin
    variables alp(n)
    maximize(-quad_form(alp,K))
    subject to
       alp >= 0;
       alp <= 1;
       Y'*alp == n/2;
cvx_end
model.alpha_all= alp;
model.dual = 1;

epsilon=1e-9;
support_v = find( alp > epsilon );
model.SVs = support_v;
model.sv_coef = alp(model.SVs);

end


function model = simplemkl_dual(K, para)
n = size(K,1);
C = para.C;
cvx_solver Gurobi
cvx_quiet true
cvx_begin
    variables alp(n)
    maximize(-0.5*quad_form(alp,K) + sum(alp))
    subject to
       alp >= 0;
       alp <= C;
       %sum(alp.*Y) == 0;
cvx_end
model.alpha_all= alp;
model.dual = 1;
epsilon=1e-9;
support_v = find( alp > epsilon );
model.SVs = support_v;
model.sv_coef = alp(model.SVs);
end