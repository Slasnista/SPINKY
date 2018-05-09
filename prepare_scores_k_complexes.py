from scipy.io import loadmat
import pandas as pd
import os

path = 'scores/k_complexes'
files = sorted(
    [os.path.join(path, f) for f in os.listdir(path)
     if "gold_standard" in f])


scores = []
for sf in files:
    print(sf)
    f = loadmat(sf)

    gs = sf.split("gold_standard")[-1].split("_")[2]

    for i in range(f["metrics_{}".format(gs)].shape[0]):
        for j in range(f["metrics_{}".format(gs)].shape[1]):
            fi = f["metrics_{}".format(gs)][i, j][0, 0]

            record = sf.split("/")[-1].split(".")[0].split("_")[-1]

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
            s["evaluation"] = "record"
            s["gold_standard"] = gs

            scores.append(s)

pd.concat(scores).to_csv("scores/spinky_2017_k_complex.csv")
