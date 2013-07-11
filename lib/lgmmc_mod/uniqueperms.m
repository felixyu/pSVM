function pu = uniqueperms(vec)
% list of all unique permutations of a vector with (possibly) replicate elements
% usage: pu = uniqueperms(vec)
% 
% arguments: (input)
% vec - 1xn or nx1 vector of elements, replicates allowed
% 
% arguments: (output)
% pu - mxn array of permutations of vec. Each row is a permutation.
%
% The result should be the same as unique(perms(vec),'rows')
% (although the order may be different.)
%
% Example:
% pu = uniqueperms([1 1 1 2 2])
% pu =
% 1 1 1 2 2
% 1 1 2 2 1
% 1 1 2 1 2
% 1 2 1 2 1
% 1 2 1 1 2
% 1 2 2 1 1
% 2 1 1 2 1
% 2 1 1 1 2
% 2 1 2 1 1
% 2 2 1 1 1
%
% See also: unique, perms
%
% Author: John D'Errico
% e-mail: woodchips@rochester.rr.com
% Release: 1.0
% Release date: 2/25/08

% How many elements in vec?
vec = vec(:); % make it always a column vector
n = length(vec);

% how many unique elements in vec?
uvec = unique(vec);
nu = length(uvec);

% any special cases?
if isempty(vec)
  pu = [];
elseif nu == 1
  % there was only one unique element, possibly replicated.
  pu = vec';
elseif n == nu
  % all the elements are unique. Just call perms
  pu = perms(vec);
else
  % 2 or more elements, at least one rep
  pu = cell(nu,1);
  for i = 1:nu
    v = vec;
    ind = find(v==uvec(i),1,'first');
    v(ind) = [];
    temp = uniqueperms(v);
    pu{i} = [repmat(uvec(i),size(temp,1),1),temp];
  end
  pu = cell2mat(pu);
end