function X = normalization(X)

% Description: With inputting the data X, this function outputs 
%              the normalized data X (each feature is nomalized to [-1,1]). 


[d,n] = size(X);

for i = 1:d
    gmax = max(X(i,:));
    gmin = min(X(i,:));
    gap = gmax - gmin;
    if gap == 0
        X(i,:) = 0;
    else
        X(i,:) = (X(i,:) - gmin)/gap;
        %X(i,:) = 2* X(i,:) -1;
    end
end

X = 2*X - 1;