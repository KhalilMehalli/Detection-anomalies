import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
from sklearn.metrics import roc_curve, auc


def list_all_file_name_in_folder(folder="../data_rapport"):
    return [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".csv")]

def csv_data_into_bunch(filenames):
    """
    This function assume that the csv file will have label egal to 1 for anomalie and 0 for normal value.
    The csv have a header too.
    """
    datasets = []
    for i in filenames:
        name = os.path.splitext(os.path.basename(i))[0]

        data = pd.read_csv(i).to_numpy()
    
        x = data[:,:-1]
        y = data[:,-1]

        # Standardisation
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Store in a Bunch
        datasets.append(Bunch(name=name, data=x_scaled, target=y))

    return datasets


def barplot(scores,y_ground_truth, color_outlier='red', color_inlier='green', title="Outliers scores " ):
    mask_outlier = (y_ground_truth==1)
    mask_inlier = ~mask_outlier

    # plt.bar need an indices tab
    indices = np.arange(scores.size)

    start = time.perf_counter()
    plt.bar(indices[mask_outlier], scores[mask_outlier], color=color_outlier, edgecolor=color_outlier)
    plt.bar(indices[mask_inlier], scores[mask_inlier], color=color_inlier, edgecolor=color_inlier)
    elapsed = time.perf_counter() - start
    print(f"Plot execution time : {elapsed:.4f} s")
    

    plt.xlabel("index points")
    plt.ylabel("Outliers scores")
    plt.title(title)


def plot_point(x,y,title):
    mask_outlier = (y==1)
    mask_inlier = ~mask_outlier
    plt.scatter(x[mask_inlier, 0], x[mask_inlier, 1], color="green", label="Inliers")
    plt.scatter(x[mask_outlier, 0], x[mask_outlier, 1], color="red", label="Outliers")
    plt.legend(bbox_to_anchor=(1.02, 1)) # Legend out of the plot if it will hide points
    plt.title(title)

def plot_roc(y_ground_truth, scores,title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_ground_truth, scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{title} (AUC = {roc_auc:.3f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.title(title)


def harmonize_labels(filenames,outlier_label='o',header=None):
    for path in filenames:
        df = pd.read_csv(path, header=header)
        
        label_col = df.columns[-1]
        
        df[label_col] = (df[label_col] == outlier_label).astype(int)
        
        df.to_csv(path, index=False)

def csv_data_infos(filenames, outlier_label=1):
    for name in filenames:
        data = pd.read_csv(name).to_numpy()
        
        x = data[:,:-1]
        y = data[:,-1]
        
        samples = x.shape[0]
        features = x.shape[1]

        # Transform label for outlier into 1 and 0 for the normal value
        y_bin = (y == outlier_label).astype(int)

        anomaly = np.sum(y_bin)
        p_anomaly = anomaly/ samples

        print({
            'filename' : name,
            'samples'  : samples,
            'features' : features,
            'anomalies': anomaly,
            'p_anomaly': p_anomaly
        })


