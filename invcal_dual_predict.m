function [ predict_response ] = invcal_dual_predict(K, model_d)

K_test_bag = zeros(size(K,1), size(model_d.alp,1)); % get the kernel matrix on the bag level
for j = 1:size(model_d.alp,1) % for each "support" bag       
    K_test_bag(:,j) = mean(K(:, model_d.bag_idx == j),2);
end
predict_response = K_test_bag * (model_d.alp - model_d.alp_2) + model_d.b;

end