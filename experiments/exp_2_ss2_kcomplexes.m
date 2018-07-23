% run spinky on spindle detection

%% Example
% Set parametres 
clear 
clc

%matlab -nodesktop -nosplash

addpath('functions');

% loop over record
data_path = '/home/infres/schambon/Papers/mcsleep/data/final_SS2/mat_files/';
fil=fullfile(data_path,'*.mat')
d=dir(fil)

epoch_length=30; 
fs=256;
thresholds = linspace(-100, 0, 21);

warning('off','all')

p = parpool('local', 20);

for k=1:numel(d)  
  file_name=fullfile(data_path,d(k).name)

  f = load(file_name)

  tr_data=data_epoching(f.chan,fs*epoch_length);

  sizes = size(f.chan);

  metrics = cell(21, 1);

  for i_th=1:numel(thresholds)  
    th = thresholds(i_th)

    % run parallel prediction
    prediction = cell(length(tr_data),1);
    parfor i=1:length(tr_data)
      [a, b] = kc_detection(double(tr_data{i}), th, fs);

      % if a == 0
      %   b = zeros(1, fs * epoch_length);
      % end
      % prediction{i} = b;

      b_ = zeros(1, fs * epoch_length);
      if a ~= 0
        for j=1:numel(b)
          b_(max((b(j) - 0.1) * fs,1):min((b(j) + 1.3) * fs, fs * epoch_length)) = 1;
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
    metrics{i_th} = compute_f1(f.kcomplex, y_pred, fs);
    metrics{i_th}.threshold = th;

  end

  a = strsplit(file_name, '/');
  b = a{1, end};
  c = strsplit(b, '.');

  save(['scores/SS2/expe_2_ss2_kcomplexes_' c{1} '.mat'], 'metrics')

end

delete(gcp('nocreate'))