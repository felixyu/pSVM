function KK = sumKbeta_lgmmc_mod(K,y_set,Sigma)

nn = size(K,1);
KK = zeros(nn);
for i = 1:length(Sigma)
    if Sigma(i) ~= 0
        KK = KK + Sigma(i)*(y_set(i,:)'*y_set(i,:)).*K;
    end
end

%KK = KK + 1/C * eye(nn);
