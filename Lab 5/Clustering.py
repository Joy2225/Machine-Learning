import cv2
import numpy as np
from scipy.fft import fft2
import pandas as pd
from datasets import load_dataset
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score


def transform(element):
    pixels = np.asarray(element["image"])
    # print(pixels.shape)
    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    fft_res = fft2(pixels).flatten()
    element["fft"] = np.abs(fft_res)
    return element


cifake_train = load_dataset(
        "dragonintelligence/CIFAKE-image-dataset", split="train[:20000]"
).map(transform, num_proc=7)
cifake_test = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="test").map(
    transform, num_proc=7
)


cifake_train.remove_columns("image")
cifake_test.remove_columns("image")

kmeans = KMeans(n_clusters=2, random_state=32).fit(cifake_train["fft"])


def kmeans_wrapper(n_clusters, X):
    kmeans = KMeans(n_clusters=n_clusters, random_state=32).fit(X)
    return {
        "labels": kmeans.labels_,
        "centres": kmeans.cluster_centers_,
        "inertia": kmeans.inertia_,
    }


def get_scores(X, start, end):
    results = {
        "Davies-Bouldin Score": [],
        "Silhouette Score": [],
        "Calinski-Harabasz Score": [],
        "distortions": [],
    }
    for i in range(start, end):
        labels, centres, inertia = kmeans_wrapper(i, X).values()
        results["Calinski-Harabasz Score"].append(calinski_harabasz_score(X, labels))
        results["Silhouette Score"].append(silhouette_score(X, labels))
        results["Davies-Bouldin Score"].append(davies_bouldin_score(X, labels))
        results["distortions"].append(inertia)
    return results


results_df = pd.DataFrame(get_scores(cifake_train["fft"], 2, 21))

results_df.to_csv("./K-Means Scores")
