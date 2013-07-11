function [cost,Alpsupaux,w0aux,posaux] = costsvmclass_lgmmc(K,y_set,StepSigma,DirSigma,Sigma,C)


global nbcall
nbcall=nbcall+1;

% nsup    = length(indsup);
% [n]=length(yapp);

Sigma = Sigma+ StepSigma * DirSigma;
%kerneloption.matrix=sumKbeta(K,Sigma);
kerneloption.matrix=sumKbeta_lgmmc_mod(K,y_set,Sigma);
[~,Alpsupaux,w0aux,posaux,~,~,cost] = svmclass_lgmmc_mod(kerneloption.matrix, C);
