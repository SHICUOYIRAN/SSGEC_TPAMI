function [id,D] = knnIndex(X, k, D_temp)
[~, nN] = size(X);
[~, idx] = sort(D_temp, 2);
id = idx(:, 2:k+1) + (nN.*[0:nN-1]'*ones(1,k));
D_temp = (D_temp + D_temp') / 2;
di = D_temp(id);
gamma= max(max(D_temp));
D = gamma * ones(nN,nN);
D(id) = di + eps;
D = D - diag(diag(D));
D = D';
end

