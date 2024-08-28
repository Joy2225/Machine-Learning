import cv2
import numpy as np
from scipy.fft import fft2
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


def transform(element):
    pixels = np.asarray(element["image"])
    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    fft_res = fft2(pixels).flatten()
    element["fft"] = np.abs(fft_res)
    return element


cifake_train = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset", split="train"
).map(transform, num_proc=7)
cifake_test = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="test").map(
    transform, num_proc=7
)


cifake_train.remove_columns("image")
cifake_test.remove_columns("image")


def extract_features(batch):
    DC_vals = [element[0] for element in batch["fft"]]
    attributes = [element[1:] for element in batch["fft"]]
    return DC_vals, attributes


def parallel_extract(dataset, num_cores=7):
    total_size = len(dataset["fft"])
    chunk_size = total_size // num_cores
    chunks = [dataset[i : i + chunk_size] for i in range(0, total_size, chunk_size)]

    DC_vals_list = []
    attributes_list = []

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(extract_features, chunk) for chunk in chunks]
        for future in as_completed(futures):
            DC_vals, attributes = future.result()
            DC_vals_list.extend(DC_vals)
            attributes_list.extend(attributes)

    return DC_vals_list, attributes_list


DC_vals_train, attributes_train = parallel_extract(cifake_train, num_cores=7)
DC_vals_test, attributes_test = parallel_extract(cifake_test, num_cores=7)

# DC_vals_train =  np.array(DC_vals_train)
# attributes_train = np.array(attributes_train)
# DC_vals_test = np.array(DC_vals_test)
# attributes_test = np.array(attributes_test)


linear_regressor = LinearRegression(n_jobs=-1).fit(attributes_train, DC_vals_train)
predictions = linear_regressor.predict(attributes_test)

results_df = pd.DataFrame(
    {"Actual DC Values": DC_vals_test, "Predicted DC Values": predictions}
)
results_df.to_csv("Predictions Of DC Values", index=False)
