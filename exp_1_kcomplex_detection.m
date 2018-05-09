% run spinky on spindle detection

%% Example
% Set parametres 
clear 
clc

addpath('functions');

% loading data under the .mat format
data_path = 'data/mat_files/';

epoch_length=30; 
fs=256;
thresholds = linspace(-100, 0, 51);


p = parpool('local', 20); 

fil=fullfile(data_path,'*.mat')
d=dir(fil)
% scores = {};
for k=1:numel(d) 

  file_name=fullfile(data_path,d(k).name)

  f = load(file_name)

  tr_data=data_epoching(f.c3,fs*epoch_length);

  sizes = size(f.c3);

  metrics_E1 = cell(51, 1);

  for i_th=1:numel(thresholds)  
    th = thresholds(i_th)

    % run parallel prediction
    prediction = cell(length(tr_data),1);
    parfor i=1:length(tr_data)
      [a, b] = kc_detection(tr_data{i}, th, fs);

      % disp(i)
      b_ = zeros(1, fs * epoch_length);
      if a ~= 0
        for j=1:numel(b)
          b_(max((b(j) - 0.1) * fs,1):min((b(j) + 1.3) * fs, 7680)) = 1;
        end
      end
      prediction{i} = b_;
    end

    % create binary vector
    y_pred = zeros(1, sizes(1, 2));
    for i=1:length(tr_data)
      y_pred((i - 1) * fs * epoch_length + 1:i * fs * epoch_length) = prediction{i};
    end

    % compute metrics over differend gold standard
    metrics_E1{i_th} = compute_f1(f.E1, y_pred, fs);
    metrics_E1{i_th}.threshold = th;

  end

  a = strsplit(file_name, '/');
  b = a{1, end};
  c = strsplit(b, '.');

  save(['scores/k_complexes/gold_standard_metrics_E1_' c{1} '.mat'], 'metrics_E1')

end
