function [ learningCurve ] = LMS1( trainInput,trainTarget,testInput,testTarget,stepSizeWeightVector )
[inputDimension,trainSize] = size(trainInput);
[aprioriErr,learningCurve,weightVector] = ...
deal(zeros(trainSize,1),zeros(1,trainSize),zeros(inputDimension,1));
% training
for n = 1:trainSize
    networkOutput = weightVector'*trainInput(:,n);
    aprioriErr(n) = trainTarget(n) - networkOutput;
    weightVector = weightVector + stepSizeWeightVector*aprioriErr(n)*trainInput(:,n);
    %testing
    err = testTarget' - (weightVector'*testInput);
    learningCurve(n) = mean(err.^2);
end

return


