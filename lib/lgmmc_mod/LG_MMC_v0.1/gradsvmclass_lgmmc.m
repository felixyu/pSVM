function [grad] = gradsvmclass_lgmmc(K,y_set,indsup,Alpsup)

% nsup  = length(indsup);
% [n] = length(yapp);

% compute the gradient of each u

if ~isstruct(K)

    d = size(y_set,1);
    grad = zeros(1,d);
    for k=1:d;
        KK = (y_set(k,:)'*y_set(k,:)).*K;
        grad(k) = - 0.5*Alpsup'*KK(indsup,indsup)*(Alpsup);
    end;
else
    %d=K.nbkernel;
    d = size(y_set,1);
    grad = zeros(1,d);
    for k=1:d;
        KK = (y_set(k,:)'*y_set(k,:)).*K;
        grad(k) = - 0.5*Alpsup'*KK*(Alpsup);
    end

end;

