function [W,S,iter2] = Dis_W(X,W,F,iddiff,dv,t)
obj2 = [0];
for iter2=1:100
    
    S = F * F';
    S(iddiff) = 0;%iddiff为KNN的索引
    isNotSymmetric = S ~= S';
    S(isNotSymmetric) = 1;
    Ds = diag(sum(S,1));
    
    
    ga=trace(W'*X*(Ds-S)*X'*W )./trace(W'*X * Ds * X'*W);
    
    QQ = X*(Ds-S)*X' - ga * X * Ds * X';
    [V, D] = eig(QQ);
    [Lam, index] = sort(diag(D));%'ascend'
    pos = length(find(abs(real(Lam)) < 1e-2)) + 1; % Finding the position of the fisrt no-zero value 
    if dv - pos + 1 < t  %如果大于0特征值对应的特征向量个数小于预定义的降维维数，则跳出循环
           flag = 0;
    break;
    end
    W = V(:, index(pos :  t + pos - 1));
    obj2 = [obj2 ga];
    if abs(obj2(iter2+1)-obj2(iter2))<1e-6 
        flag = 1;
        break;
    end
    iter2=iter2+1;
end
% plot(obj2)
end