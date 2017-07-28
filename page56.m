clear all,
close all
clc
%% filter config
%{
% the note can be fold
% time delay (embedding) length
%}
[inputDimension,a,np,trainSize,testSize] = deal(10,1,0.04,500,100);
disp('Learning curves are generating. Please wait...');
%% data formatting
load MK30 %MK30 5000*1
MK30 = MK30 + np*randn(size(MK30));
MK30 = MK30 - mean(MK30);
[train_set,test_set] = deal(MK30(1501:4500),MK30(4601:4900));%3000,300

[trainInput,trainTarget,testInput,testTarget] = deal(...
    zeros(inputDimension,trainSize),...
    train_set(inputDimension+1:inputDimension+trainSize),...
    zeros(inputDimension,testSize),...
    test_set(inputDimension+1:inputDimension+testSize));
% data embedding
for k=1:trainSize
	trainInput(:,k) = train_set(k:k+inputDimension-1);
end
% Test data
% the test data is after the training data 
for k=1:testSize
	testInput(:,k) = test_set(k:k+inputDimension-1);
end
%===end of data===
%% LMS
[ learningCurve ] = LMS1( trainInput,trainTarget,testInput,testTarget,.2);
%% KLMS
 [ ~,mse_te_k ] =...
KLMS1( trainInput,trainTarget,testInput,testTarget,'Gauss',a,.2);
%% plot
%===plot and test===
figure
plot(learningCurve,'r-','LineWidth',2);
hold on
plot(mse_te_k,'b--','LineWidth',2);
grid on
set(gca,'FontSize',14)
set(gca,'FontName','Arial');
legend('LMS','KLMS')
xlabel('Iteration')
ylabel('MSE')
%===end of plot and test===
