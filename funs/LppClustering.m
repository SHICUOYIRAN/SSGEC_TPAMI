function [S, F, W, time, iter,flag] = LppClustering(X, gt, t, gamma, resultFile)
% min \sum_{i,j}{||W'(x_i - x_j)||_F^2 * s_ij}
% s.t. W'W = I, F\in Index, WXDX'W'=I, 
% 如果x_i, x_j为k近邻点，s_ij = <f_i, f_j>，否则 s_ij = 0;


tic

[nN, dv] = size(X');
nC = length(unique(gt));


% Init F
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
err_W = [];
while(1)
    % solve W
    S = F * F';
    Ds = diag(sum(S,1));
    [V, D] = eig(X*(Ds-S)*X'- gamma* X*Ds*X');
    [Lam, index] = sort(diag(D));%'descend'
    pos = length(find(abs(real(Lam)) < 1e-2)) + 1; % Finding the position of the fisrt no-zero value 
    if dv - pos + 1 < t
         flag = 0;
        break;
    end
%     W = V(:, index(pos : min(dv, t + pos - 1)));
 W = V(:, index(pos :  t + pos - 1));
    
    % solve F
    Dw = L2_distance_1(W'*X, W'*X);
    iter_Y = 1;
    obj_Y = [0];
    while(1)
        F = Dis_Y(Dw,F);
        %Converge
        obj_Y = [obj_Y trace(trace(F'*Dw*F))];
        if abs(obj_Y(iter_Y+1)-obj_Y(iter_Y))<1e-6 || iter_Y > 20
            break;
        end
        iter_Y = iter_Y + 1;
    end

    % Is converge
    obj = [obj trace(F'*Dw*F)];
    if abs(obj(iter+1)-obj(iter))<1e-6 || iter > 200
         flag = 1;
        break;
    end


    % W_pre = imresize(W_pre, size(W));
    % err_W = [err_W abs(sum(W_pre(:)) - sum(W(:)))];
    % W_pre = W;
    % if err_W(iter) < 1e-3 || iter > 100
    %     break;
    % end
    iter = iter + 1;
end

time = toc;


