function [x,distance_matrix] = gaussian_kernel(x,ratio)

%Description: With inputting the data x and the parameter ratio of gaussian
%             kernel, this function outputs the kernel matrix x and 
%             the distance matrix distance_matrix. 

% X: d*n

[d,n] = size(x);
distance_matrix = repmat(sum(x'.*x',2)',n,1) + repmat(sum(x'.*x',2),1,n) ...
            - 2*x'*x;

sigma_mean = sum(sum(distance_matrix))/n^2;

% sigma_mean = sum(sum(repmat(sum(x'.*x',2)',n,1) + repmat(sum(x'.*x',2),1,n) ...
%             - 2*x'*x))/n^2;
        
sigma_mean = sigma_mean^0.5;

sigma = ratio * sigma_mean;

x = exp(-distance_matrix/(2*sigma^2));


