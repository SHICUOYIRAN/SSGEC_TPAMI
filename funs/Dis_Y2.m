function [Y,y_ind] = Dis_Y2(D,Y_pre,H, lambda)
nIter = 1000;
nN = size(Y_pre, 1);
n_clu = sum(Y_pre, 1);
[~, y_ind] = max(Y_pre, [], 2);
yD2_H =  2 * D' * Y_pre - lambda * H;
y_pre = y_ind;
yp=[];
for iter = 1 : nIter
    for i = 1:nN 
        p = y_ind(i); 
        if n_clu(p) == 1 
            continue;
        end
        [~, r] = min(yD2_H(i, :));
                if r ~= p
                    n_clu(p) = n_clu(p)-1;
               n_clu(r) = n_clu(r)+1;
            yD2_H(:, r) = yD2_H(:, r) + 2 * D(i, :)';
            yD2_H(:, p) = yD2_H(:, p) - 2 * D(i, :)';
            y_ind(i) = r;
                end
        Y = full(ind2vec(y_ind'))';        
    end
err = sum(y_pre ~= y_ind);
if err == 0 
break;
end
yp = [yp err];
y_pre = y_ind;
end
Y = full(ind2vec(y_ind'))';
end