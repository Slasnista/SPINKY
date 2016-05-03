function [sp_optimal_thresh] = sp_train(sig,sp_thresh,fs,sp_expert_score,figure_set)
n=1;
for kk=sp_thresh;
   for i=1:length(sig)
      %spindles detection
        [nbr_sp(i),pos_sp] = sp_detection(sig{i},kk,fs);
   end
   
   [sp_Sen(n),sp_FDR(n)] = performances_measure(sp_expert_score,nbr_sp);% the user can use his own function to compute sensitivity and FDR
   n=n+1;
end
%optimal threshold selection: the one which maximize the difference between
% Sen and FDR
[sp_optimal_thresh]=ROC_curve(sp_FDR,sp_Sen,sp_thresh,figure_set);

end

