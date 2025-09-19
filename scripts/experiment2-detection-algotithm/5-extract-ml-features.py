import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algorithm import moving_average, moving_median
from scipy.stats import skew, kurtosis

dataset = np.load("scripts/experiment2-detection-algotithm/data/signal_dataset.npy")
annotation_enter = np.load("scripts/experiment2-detection-algotithm/data/annotation_enter.npy")
annotation_leave = np.load("scripts/experiment2-detection-algotithm/data/annotation_leave.npy")

feature_list = ["mean", "std_dev", "min_val", "max_val", "energy", "skewness", "kurtosis"]

df = pd.DataFrame(columns=feature_list + ["enter_label", "leave_label"])

for chunk, enter, leave in zip(dataset, annotation_enter, annotation_leave):
    mean = np.mean(chunk)
    std_dev = np.std(chunk)
    min_val = np.min(chunk)
    max_val = np.max(chunk)
    energy = np.sum(chunk**2)
    skewness = skew(chunk)
    kurt = kurtosis(chunk)
    row = [mean, std_dev, min_val, max_val, energy, skewness, kurt, enter, leave]
    df.loc[len(df)] = row

print(df)

df.to_csv("scripts/experiment2-detection-algotithm/data/feature_dataset.csv", index=False)