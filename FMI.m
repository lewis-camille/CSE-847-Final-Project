
% Clustering performance evaluation against ground truth
% Fowlkes-Mallows Index (FMI)
function score = FMI(pred_labels, true_labels)
% INPUT
% pred_labels : vector of labels
% true_labels : vector of labels
%
% OUTPUT
% score       : measures level of agreement between the two input 
%               assignments; the integers in each set do not have to 
%               correspond to the same clusters

dimensions = size(pred_labels);
N = dimensions(1);
% Count TP, FP, FN
TP = 0;
FP = 0;
FN = 0;
% Go through pairs of points
for i = 1:N-1
    for j = i:N
        pred1 = pred_labels(i);
        pred2 = pred_labels(j);
        true1 = true_labels(i);
        true2 = true_labels(j);
        if pred1 == pred2 && true1 == true2
            TP = TP + 1;
        elseif pred1 ~= pred2 && true1 == true2
            FP = FP + 1;
        elseif pred1 == pred2 && true1 ~= true2
            FN = FN + 1;
        end
    end
end
% Compute FMI
score = TP ./ (sqrt((TP + FP) .* (TP + FN)));
end