function init
if ~exist('INIT', 'var')
run ../../../felix_codebase/matlab_lib/CVX/cvx_setup
addpath (genpath('./lib/lgmmc_mod/'));
addpath ../../../felix_codebase/matlab_lib/classification/libsvm/matlab_parallel/
%addpath ../../../felix_codebase/matlab_lib/classification/liblinear/matlab/
%addpath ../../../felix_codebase/matlab_lib/classification/libsvm_nonparallel/matlab/
%addpath ../../../felix_codebase/matlab_lib/classification/libsvm/matlab/
cvx_solver Gurobi % in order to run on the cluster
cvx_quiet true
addpath('lib')
INIT = 1;
end
end
