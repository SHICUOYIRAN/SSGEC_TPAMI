function [F] = KnnLppClustering(X, gt, t, k, gamma,lambda,p)
[nN, dv] = size(X');
nC = length(unique(gt));
Do = L2_distance_1(X,X);
[id, ~] = knnIndex(X, k, Do);
iddiff = setdiff(1:nN^2, id);
rng(1);
prelabel = kmeans(X', nC) + (0:nN-1)'.*nC;
F = zeros(nC, nN);
F(prelabel) = 1;
F = F';
if t > dv
    error("降维后的维度大于原始维度");
end
W = eye(dv, t);

%% ================= Optimizate =====================
iter = 1;
obj = [0];
while(1)
    S = F * F';
    S(iddiff) = 0;
    isNotSymmetric = S ~= S';
    S(isNotSymmetric) = 1;
    Ds = diag(sum(S,1));
    QQ = X*(Ds-S)*X' - gamma * X * Ds * X';
    [V, D] = eig(QQ);
    [Lam, index] = sort(diag(D));
    pos = length(find(abs(real(Lam)) < 1e-2)) + 1; 
    if dv - pos + 1 < t  
           flag = 0;
    break;
    end
    W = V(:, index(pos :  t + pos - 1));
    SSS = diag(1./sqrt(sum(F.^2)));
    H=F*SSS; 
    Dw = L2_distance_1(W'*X, W'*X); 
    F = Dis_Y2(Dw,F,H, lambda);
    obj = [obj (trace(F'*Dw*F)-lambda*sum(sqrt(sum(F.^2))))];
    if abs(obj(iter+1)-obj(iter))<1e-6 || iter > 1000
        flag = 1;
        break;
    end
    iter = iter + 1;
end


