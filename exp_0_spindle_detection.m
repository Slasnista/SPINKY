% run spinky on spindle detection

%% Example
% Set parametres 
clear 
clc

addpath('functions');
addpath('../mcsleep/');

%matlab -nodisplay -nosplash
data_path = '/home/infres/schambon/Papers/mcsleep/data/mat_gold_standard/';

epoch_length=30; 
fs=256;
thresholds = linspace(0, 250, 6);


p = parpool('local', 20); 

fil=fullfile(data_path,'*.mat')
d=dir(fil)
% scores = {};
for k=1:numel(d)  
  file_name=fullfile(data_path,d(k).name)

  f = load(file_name)

  tr_data=data_epoching(f.c3,fs*epoch_length);

  sizes = size(f.c3);

  metrics_E1 = cell(5, 1);
  metrics_E2 = cell(5, 1);
  metrics_union = cell(5, 1);
  metrics_intersection = cell(5, 1);

  for i_th=2:numel(thresholds)  
    th = thresholds(i_th)

    % run parallel prediction
    prediction = cell(length(tr_data),1);
    parfor i=1:length(tr_data)
      [a, b] = sp_detection(tr_data{i}, th, fs);

      if a == 0
        b = zeros(1, fs * epoch_length);
      end
      prediction{i} = b;
    end

    % create binary vector
    y_pred = zeros(1, sizes(1, 2));
    for i=1:length(tr_data)
      y_pred((i - 1) * fs * epoch_length + 1:i * fs * epoch_length) = prediction{i};
    end

    % compute metrics over differend gold standard
    metrics_E1{i_th - 1} = compute_f1(f.E1, y_pred, fs);
    metrics_E1{i_th - 1}.threshold = th;

    metrics_E2{i_th - 1} = compute_f1(f.E2, y_pred, fs);
    metrics_E2{i_th - 1}.threshold = th;

    metrics_union{i_th - 1} = compute_f1(f.Union, y_pred, fs);
    metrics_union{i_th - 1}.threshold = th;

    metrics_intersection{i_th - 1} = compute_f1(f.Intersection, y_pred, fs);
    metrics_intersection{i_th - 1}.threshold = th;

  end

  a = strsplit(file_name, '/');
  b = a{1, end};
  c = strsplit(b, '.');

  save(['scores/spindles/gold_standard_metrics_E1_' c{1} '.mat'], 'metrics_E1')
  save(['scores/spindles/gold_standard_metrics_E2_' c{1} '.mat'], 'metrics_E2')
  save(['scores/spindles/gold_standard_metrics_union_' c{1} '.mat'], 'metrics_union')
  save(['scores/spindles/gold_standard_metrics_intersection_' c{1} '.mat'], 'metrics_intersection')

end
