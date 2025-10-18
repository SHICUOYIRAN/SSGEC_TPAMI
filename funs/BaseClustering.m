function [dist] = BaseClustering(X, gt)
[nN, dv] = size(X');
nC = length(unique(gt));

rng(1);
prelabel = kmeans(X', nC) + (0:nN-1)'.*nC;
F = zeros(nC, nN);
F(prelabel) = 1;
F = F';

Dw = L2_distance_1(X, X);
iter_Y = 1;
obj_Y = [0];
while(1)
    F = Dis_Y(Dw,F);
    %Converge
    obj_Y = [obj_Y trace(trace(F'*Dw*F))];
    if abs(obj_Y(iter_Y+1)-obj_Y(iter_Y))<1e-6 || iter_Y > 200
        break;
    end
    iter_Y = iter_Y + 1;
end

[~, label] = max(F');
result = ClusteringMeasure(gt, label);

fprintf('base：%.4f,%.4f,%.4f \n',result(1:3))

end