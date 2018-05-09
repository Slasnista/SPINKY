function iou = compute_iou(true_starts, true_ends, pred_starts, pred_ends)

n_preds = size(pred_starts);
n_preds = n_preds(1, 2);

n_trues = size(true_starts);
n_trues = n_trues(1, 2);


true_starts = transpose(true_starts);
true_ends = transpose(true_ends);

true_starts = repmat(true_starts, [1, n_preds]);
true_ends = repmat(true_ends, [1, n_preds]);

pred_starts = repmat(pred_starts, [n_trues, 1]);
pred_ends = repmat(pred_ends, [n_trues, 1]);

% get the element wise maximum of starts
x_a = max(true_starts, pred_starts);

% get the minimu wise minium of ends
x_b = min(true_ends, pred_ends);

% get inter area
inter_area = (x_b - x_a);

% get respective areas
area_a = true_ends - true_starts;
area_b = pred_ends - pred_starts;

iou = inter_area ./ (area_a + area_b - inter_area);

end
