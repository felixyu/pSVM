function [obj_decrease, idx_sorted] = lib_compute_flip( delta, min_po, max_po )
%LIB_COMPUTE_FLIP Summary of this function goes here
%   Detailed explanation goes here

% y_i are initlized as -1
% by flipping the sign of y_i to 1, the objective will decrease delta_i
% we must have min_po <= #positive <= max_po
% output a vector with dimension max_po - min_po + 1


[delta_sorted, idx_sorted] = sort(delta, 'descend');
obj_decrease = [0; cumsum(delta_sorted)];
obj_decrease = obj_decrease(min_po+1: max_po+1);

end

