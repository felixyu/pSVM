function [ y ] = recover_y(K0, beta, y_set)
%RECOVER_Y Summary of this function goes here
%   Detailed explanation goes here
n = size(K0,1);
ind = find(beta > eps);
M = zeros(n,n);
for i = 1:length(ind)
    M = M + beta(ind(i))*   (K0.*(y_set(ind(i),:)'*y_set(ind(i),:)));
end

K0(K0 == 0) = eps;
%try
[V,D] = eigs(M./K0,1); % I am not sure whether SVD will perserve the label propotions
%catch
%opts.tol = 1e-3;
%[V,D] = eigs(M./K0,1,'lm',opts);
%end

% such that V*D*V' \approximately= M./K0
y = sign(V*D)';
end