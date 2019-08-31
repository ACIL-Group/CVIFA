%% """ PBM validity index """
%
% PROGRAM DESCRIPTION
% This is a MATLAB implementation of the PBM validity index.
%
% REFERENCES:
% [1] M. K. Pakhira, S. Bandyopadhyay, and U. Maulik, “Validity index for
% crisp and fuzzy clusters,” Pattern Recognition, vol. 37, no. 3, pp. 487 –
% 501, 2004.
%
% Code written by Leonardo Enzo Brito da Silva
% Under the supervision of Dr. Donald C. Wunsch II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function PBM = PBM_index(x, labels) 

    u = unique(labels);
    nClusters = length(u);
    dim = size(x, 2);
    zi = zeros(nClusters, dim);

    z1 = mean(x, 1);
    D1 = pdist2(x, z1, 'euclidean');
    E1 = sum(D1);

    Ek = 0;
    for i=1:nClusters
        xi = x(labels == u(i), :);
        zi(i,:) = mean(xi, 1);
        Di = pdist2(xi, zi(i,:), 'euclidean');
        Ek = Ek + sum(Di);
    end

    Dk = max(pdist(zi));

    PBM = ((1/nClusters)*(E1/Ek)*Dk)^2;

end    