function [ expansionCoefficient,mse_te_k ] =...
KLMS1( trainInput,trainTarget,testInput,testTarget,ker_type,ker_param,stepSizeFeatureVector)
% use function [ y ] = ker_eval( X1,X2,ker_type,ker_param )
%% init
% memeory initialization
[trainSize,testSize] = deal(length(trainTarget),length(testTarget));
%{
    e_k = zeros(trainSize,1);
    expansionCoefficient = zeros(trainSize,1);
    networkOutput = zeros(trainSize,1);
    mse_te_k = zeros(trainSize,1);
%}
[e_k,expansionCoefficient,networkOutput,mse_te_k] = deal(zeros(trainSize,1));
% n=1 init
[e_k(1),networkOutput(1),mse_te_k(1)] = deal(testTarget(1),0,mean(testTarget.^2));
%% start
for n = 2:trainSize
    %training
    %filtering
    ii = 1:n-1;
    networkOutput(n) = expansionCoefficient(ii)'*ker_eval(trainInput(:,n),trainInput(:,ii),ker_type,ker_param);
    e_k(n) = trainTarget(n) - networkOutput(n);
%     % updating
%     weightVector =  weightVector + stepSizeWeightVector*aprioriErr*trainInput(:,n);
%     biasTerm = biasTerm + stepSizeBias*aprioriErr;
    expansionCoefficient(n) = stepSizeFeatureVector*e_k(n);
    
    networkOutputTest = zeros(testSize,1);
    for jj = 1:testSize
    networkOutputTest(jj) = expansionCoefficient(1:n)'*ker_eval(testInput(:,jj),trainInput(:,1:n),ker_type,ker_param);
    end
    
    err = testTarget - networkOutputTest;  
    mse_te_k(n) = mean(err.^2);
    
end

return
