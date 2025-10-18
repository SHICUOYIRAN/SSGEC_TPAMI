function [id] = knnIndex2(X, k)
%根据欧式距离选择每个样本的k近邻点并用1标记位置
% X ： d x n
[~, nN] = size(X);
D_temp = L2_distance_1(X,X);
gamma= max(max(D_temp));
[~, idx] = sort(D_temp, 2);
D = zeros(nN,nN);
id = idx(:, 2:k+1) + (nN.*[0:nN-1]'*ones(1,k));
% di = D_temp(id);
% D(id) = 1;
% D = D - diag(diag(D));
% D = D';
end