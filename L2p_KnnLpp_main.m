clear
close all;
addpath('.\funs');
addpath('.\datasets');
dataname ='ORL';  gamma= 0.0008;k= 5;dim=375;lambda=60000;

load([dataname '.mat']);
[n, dv] = size(X);
c = length(unique(Y));
Y = double(Y);
X = double(X);

[F] = KnnLppClustering(X',Y, dim, k, gamma,lambda,1); 
[~, label] = max(F');
result = ClusteringMeasure(Y, label)
