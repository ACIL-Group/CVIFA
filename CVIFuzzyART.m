%% """ CVI-based FA """
% 
% PROGRAM DESCRIPTION
% This is a MATLAB implementation of the "Cluster Validity Index-based Vigilance Test in ART" using Fuzzy ART network type.
%
% REFERENCES:
% [1] L. E. Brito da Silva and Donald C. Wunsch II, “Validity Index-based Vigilance 
% Test in Adaptive Resonance Theory Neural Networks,” in 2017 IEEE Symposium on
% Computational Intelligence and Data Mining (CIDM), 2017.
% [2] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast 
% stable learning and categorization of analog patterns by an adaptive 
% resonance system," Neural networks, vol. 4, no. 6, pp. 759–771, 1991.
% [3] T. Calinski and J. Harabasz, “A dendrite method for cluster analysis,”
% Communications in Statistics, vol. 3, no. 1, pp. 1–27, 1974.
% [4] M. K. Pakhira, S. Bandyopadhyay, and U. Maulik, “Validity index for
% crisp and fuzzy clusters,” Pattern Recognition, vol. 37, no. 3, pp. 487 –
% 501, 2004.
% [5] D. L. Davies and D. W. Bouldin, “A cluster separation measure,” IEEE
% Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-
% 1, no. 2, pp. 224–227, April 1979.
% [6] P. J. Rousseeuw, “Silhouettes: A graphical aid to the interpretation and
% validation of cluster analysis,” Journal of Computational and Applied
% Mathematics, vol. 20, pp. 53 – 65, 1987.
%
% Code written by Leonardo Enzo Brito da Silva
% Under the supervision of Dr. Donald C. Wunsch II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fuzzy ART Class
classdef CVIFuzzyART    
    properties
        rho                 % vigilance parameter: [0,1]  
        alpha = 1e-7;       % choice parameter (alpha = 1e-7: suggested in the article)
        beta = 1;           % learning parameter: ]0,1] (beta=1: "fast learning" suggested in the article)        
        W = [];             % top-down weights 
        labels = [];        % best matching units (class labels)
        dim = [];           % original dimension of data set  
        nCat = 0;           % total number of categories
        Epoch = 0;          % current epoch
    end   
    methods        
        % Assign property values from within the class constructor
        function obj = CVIFuzzyART(settings) 
            obj.rho = settings.rho;
            obj.alpha = settings.alpha;
            obj.beta = settings.beta;
        end         
        % Train
        function obj = train(obj, data, maxEpochs, valind)            
            %% Data Information            
            [nSamples, obj.dim] = size(data);
            obj.labels = zeros(nSamples, 1);
            
            %% Complement Coding
            x = CVIFuzzyART.complement_coder(data);
            if isempty(valind)
                valind = '';
            end
            
            %% First Category Initialization           
            if isempty(obj.W)
                obj.W = x(1,:);                
                obj.labels(1) = 1;
                obj.nCat = 1;
                sample_no1 = 2;
            else
                sample_no1 = 1;
            end    
            W_old = obj.W;
            labels_old = obj.labels;            
            %% Learning
            obj.Epoch = 0;
            vu = [false false];
            while(true)
                obj.Epoch = obj.Epoch + 1;
                for i=sample_no1:nSamples %loop over samples   
                    % Compute Activation Function 
                    T = zeros(obj.nCat, 1);                    
                    for j=1:obj.nCat    
                        T(j,1) = activation(obj, x(i,:), j); % category choice
                    end
                    % Sort activation function values in descending order
                    [~, index] = sort(T, 'descend'); % WTA
                    % Mismatch Flag
                    mismatch_flag = true; 
                    for j=1:obj.nCat % For the number of categories
                        % Best Matching Unit
                        bmu = index(j);                        
                        % Compute Match Function 
                        M = match(obj, x(i,:), bmu);
                        vu(1) = M >= obj.rho;
                        if vu(1) % Vigilance Check - Pass vu(1)                            
                            % "Match Tracking" Procedure 
                            new_labels = obj.labels;
                            new_labels(i) = bmu;  
                            if strcmp(valind, 'PBM')
                                VI_old = PBM_index(data, obj.labels+1);
                                VI_new = PBM_index(data, new_labels+1);
                            else
                                if ~strcmp(valind, '')
                                    eva = evalclusters(data, obj.labels+1, valind);
                                    VI_old = eva.CriterionValues; 
                                    eva = evalclusters(data, new_labels+1, valind);
                                    VI_new = eva.CriterionValues;
                                end
                            end                            
                            switch valind  % Compute Validity indices in R^dim space of data; NOT in R^{2*dim} space of x
                                case 'CalinskiHarabasz'                                       
                                    vu(2) = VI_new >= VI_old;
                                case 'DaviesBouldin'
                                    vu(2) = VI_new <= VI_old;
                                case 'silhouette'
                                    vu(2) = VI_new >= VI_old;
                                case 'PBM'
                                    vu(2) = VI_new >= VI_old;
                                otherwise
                                    vu(2) = true;
                            end 
                            % Learning
                            if vu(2)  % Vigilance Check - Pass vu(2)                               
                                obj = learn(obj, x(i,:), bmu);
                                obj.labels(i) = bmu;
                                mismatch_flag = false;
                                break; 
                            end                            
                        end                               
                    end  
                    if mismatch_flag
                        obj.W(end+1,:) = x(i,:);                                
                        obj.labels(i) = size(obj.W, 1);
                        obj.nCat = obj.nCat + 1;
                    end 
                clc; fprintf('Epoch: %d \nSample ID: %d \nnCat: %d \n', obj.Epoch, i, obj.nCat);  %display training info
                end  
                sample_no1 = 1; % Start loop from 1st sample from 2nd epoch and onwards
                % Stopping Criteria
                if isequal(obj.W, W_old)
                    break;
                end       
                if isequal(obj.labels, labels_old)
                    break;
                end
                if obj.Epoch >= maxEpochs
                    break;
                end 
                W_old = obj.W;
                labels_old = obj.labels; 
            end            
        end 
        % Activation Function
        function T = activation(obj, x, index)
            T = norm(min(x, obj.W(index,:)),1)/(obj.alpha + norm(obj.W(index,:), 1));            
        end  
        % Match Function
        function M = match(obj, x, index)
            M = norm(min(x, obj.W(index,:)),1)/obj.dim;           
        end  
        % Learning
        function obj = learn(obj, x, index)
            obj.W(index,:) = obj.beta*(min(x, obj.W(index,:))) + (1-obj.beta)*obj.W(index,:);         
        end   
    end    
    methods(Static)
        % Linear Normalization and Complement Coding
        function x = complement_coder(data)
            x = mapminmax(data', 0, 1);
            x = x';
            x = [x 1-x];
        end         
    end
end