function [y,ct]= lgmmc(x,opt)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: [y,ct]= lgmmc(x,opt)
% With inputting the data x and some options opt, this function outputs the
% predicted label vector y and the cputime ct. 

% input:
% x: d*n
% opt
% output:
% y: n*1
% ct: cputime

ttt = cputime;

%%%%%%%%%%%%%%%%%%%%%%%%
options.name = 'null';
options.norm = 1;
options.gaussian = 0;
options.ratio = 1;
options.C = 1;
options.iteration = 20;
options.bias = 1;
options.address = 'null';

if isfield(opt,'name');
    options.name = opt.name;
end

if isfield(opt,'norm');
    options.norm = opt.norm;
end

if isfield(opt,'C');
    options.C = opt.C;
end

if isfield(opt,'gaussian');
    options.gaussian = opt.gaussian;
end

if isfield(opt,'ratio');
    options.ratio = opt.ratio;
end

if isfield(opt,'iteration');
    options.iteration = opt.iteration;
end

if isfield(opt,'bias');
    options.bias = opt.bias;
end

if isfield(opt,'address');
    options.address = opt.address;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[d,n] = size(x);

if options.norm == 1
   disp('normalizing data...');
   x = normalization(x); 
end

if options.gaussian == 1
   disp('calculating the kernel...');
   K0 = gaussian_kernel(x,options.ratio);
else
   K0 = x'*x; 
end

if options.bias == 1 
    K0 = K0 + ones(n);
end


if options.gaussian == 1
    [U,A] = svd(K0);
    s = sqrt(diag(A));
    x_new = U*diag(s)*U';
    clear U A;
else
    x_new = x;
end


[y] = LGMMC_train(K0,x_new,options.C,opt.ep,options.iteration);

clear x K0 x_new;

ct = cputime - ttt;

