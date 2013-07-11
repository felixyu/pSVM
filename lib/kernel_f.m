function [ K ] = kernel_f(data, method)
%KERNEL_F Summary of this function goes here
%   Detailed explanation goes here
switch(method)
    case 'linear'
        K = data*data';
    case 'rbf1'
        K = pdist2(data, data);
        K = exp(-1*K);
    case 'rbf01'
        K = pdist2(data, data);
        K = exp(-0.1*K);
    case 'rbf001'
        K = pdist2(data, data);
        K = exp(-0.01*K);
end
end