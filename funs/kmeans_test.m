function kmeans_test(X, gt)
nC = length(unique(gt));
label = kmeans(X, nC);
res = ClusteringMeasure(gt, label);
disp(res);
end
