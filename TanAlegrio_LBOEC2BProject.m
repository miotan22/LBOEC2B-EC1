close all; clear; clc

%% Import assets
T = readtable('weather.csv');
dataT = T(:,[1:5 7 10:19 21]);

%% Data Preparation
data = dataT{:,:};
input = data(:,[1:16])';
target = data(:,17)';

[input_norm, mui, sdi] = zscore(input);
[target_norm, mut, sdt] = zscore(target);

%% Neural Network Preparation
rng('default');
hiddenSizes =  [36 35 24 13 8 5];

myNetwork = fitnet(hiddenSizes, 'trainlm');

myNetwork.divideParam.trainRatio = 0.8;
myNetwork.divideParam.valRatio = 0.1;
myNetwork.divideParam.testRatio = 0.1;

[myNetwork, tr] = train(myNetwork, input_norm, target_norm);
view(myNetwork);
output = myNetwork(input_norm(:,[1:100]));

rainfall = dataT{:,3}';

output_denorm = (output*sdt) + mut;

figure(1);

subplot(2,1,1); plot(1:100, rainfall(:,1:100));
xlim([0 100]);
xlabel('Days');
ylabel('Rainfall (mm)');
title('Rainfall For Today');
subplot(2,1,2); plot (2:101, output_denorm(:,1:100),'r--');
xlim([0 101]);
xlabel('Days');
ylabel('Rainfall (mm)');
title('Rainfall Prediction');

figure(2);
bar(1:100, rainfall(:,1:100));
hold on;
plot(2:101, output_denorm(:,1:100),'r--', 'LineWidth', 1);
xlim([0 101]);
xlabel('Days');
ylabel('Rainfall (mm)');
title('Rainfall Prediction');
legend('Rainfall(today)', 'Predicted Rainfall');

