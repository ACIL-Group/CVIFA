% This is an example of usage of Validity Index-based Vigilance Test in Fuzzy ART 
%
% PROGRAM DESCRIPTION
% This program exemplifies the usage of the CVI Fuzzy ART code provided.
%
% REFERENCES:
% [1] L. E. Brito da Silva and Donald C. Wunsch II, “Validity Index-based Vigilance 
% Test in Adaptive Resonance Theory Neural Networks,” in 2017 IEEE Symposium on
% Computational Intelligence and Data Mining (CIDM), 2017.
%
% Code written by Leonardo Enzo Brito da Silva
% Under the supervision of Dr. Donald C. Wunsch II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Clean Run
clear variables
close all
echo off
clc

%% Load data
load('clusterdemo.dat')
data = clusterdemo;
[nSamples, dim] = size(data);
% Linear Normalization
data = mapminmax(data', 0, 1);
data = data';
% Randomize Presentation
P = randperm(nSamples);
data = data(P, :);
%% Adaptive Resonance Theory
% Fuzzy ART Parameters
settings = struct();
settings.rho = 0;
settings.alpha = 1e-3;
settings.beta = 1;
nEpochs = 20;
CVI = 'CalinskiHarabasz';
% OPTIONS: 
% 'CalinskiHarabasz', 'DaviesBouldin', 'silhouette', 'PBM' or empty []. 
% If empty [] then a standard Fuzzy ART will be trained.
% Train Fuzzy ART
FA = CVIFuzzyART(settings);
FA = FA.train(data, nEpochs, CVI);
%% Plot clustering results
siz = 20;
clrs = rand(FA.nCat, 3);
C = zeros(nSamples, 3);
for k=1:nSamples
    C(k,:) = clrs(FA.labels(k),:);
end
figure
scatter3(data(:,1), data(:,2), data(:,3), siz, C, 'filled')
grid on
box on
axis square