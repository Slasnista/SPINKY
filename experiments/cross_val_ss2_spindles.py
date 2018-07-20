import os
import pandas as pd
from scipy.io import loadmat

path = "scores/SS2"
files = [os.path.join(path, s) for s in os.listdir(path) if "spindles" in s]

scores = []
for sf in files:
    pass

    f = loadmat(sf)

    record = sf.split("_")[-1].split(".")[0]

    for i in range(f["metrics"].shape[0]):
        for j in range(f["metrics"].shape[1]):
            fi = f["metrics"][i, j][0, 0]

            precision = fi[0]
            recall = fi[1]
            f1 = fi[2]
            IoU = fi[3]
            by_sample_precision = fi[4]
            by_sample_recall = fi[5]
            by_sample_f1 = fi[6]
            threshold = fi[7]
            # threshold = fi[8][0]

            s = pd.DataFrame()
            s["precision"] = precision.squeeze()
            s["recall"] = recall.squeeze()
            s["f1"] = f1.squeeze()
            s["IoU"] = IoU.squeeze()
            s["by_sample_precision"] = by_sample_precision[0, 0]
            s["by_sample_recall"] = by_sample_recall[0, 0]
            s["by_sample_f1"] = by_sample_f1[0, 0]
            s["threshold"] = threshold.squeeze()
            s["record"] = record

            scores.append(s)

scores = pd.concat(scores)
scores = scores.fillna(0)

records = sorted(pd.unique(scores.record))

# cross valition
final_scores = []
for idx_split in range(5):

    # record for testing
    r_test = records[4 * idx_split: 4 * (idx_split + 1)]

    # record for validation
    r_ = [r for r in records if r not in r_test]

    s_val = scores[scores.record.isin(r_)]
    s_test = scores[scores.record.isin(r_test)]

    # hp selection
    s_val = s_val.groupby(["IoU", "threshold"]).mean().reset_index()

    for IoU in pd.unique(s_val.IoU):

        f1_max = s_val[s_val.IoU == IoU]["f1"].max()
        s_val_max = s_val[(s_val.IoU == IoU) & (s_val.f1 == f1_max)]
        threshold = s_val_max.threshold.values[0]

        s_test_ = s_test[
            (s_test.IoU == IoU) &
            (s_test.threshold == threshold)].copy()

        final_scores.append(s_test_)

final_scores = pd.concat(final_scores)

final_scores.to_csv("scores/SS2/final_scores_spindles.csv")
