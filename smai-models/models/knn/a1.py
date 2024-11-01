from knn_optimized import KNN_model
import pandas as pd
import numpy as np
import time
from performance import performanceMetrics
import matplotlib.pyplot as plt

## preprocessing steps of data (dropping unecessary and duplicate columns)
def preprocess():
    df = pd.read_csv("../../data/external/spotify.csv")
    df.drop_duplicates(subset='track_name', inplace=True)
    columns_to_drop = ['s_no','track_id','track_name', 'album_name', 'artists', 'time_signature', 'key', 'mode']
    df.drop(columns_to_drop, axis=1, inplace=True)


    # Normalize columns with integer or float values using Gaussian normalization
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std

    # for col in numeric_columns:
    #     min_val = df[col].min()
    #     max_val = df[col].max()
    #     df[col] = (df[col] - min_val) / (max_val - min_val)

    # Convert boolean columns to integers
    bool_columns = df.select_dtypes(include=bool).columns
    df[bool_columns] = df[bool_columns].astype(int)

    ## duration column values get fucked because of this, have to change it.
    ## normalize by fitting to exponential or something instead of between max and min.

    # df.to_csv("../../data/interim/spotify_preprocessed_gaussian.csv", index=False)
    # print("HI")
    return df

def drop_random_columns(df, n):
    # Ensure n is not greater than the number of columns
    if n > df.shape[1]:
        raise ValueError("n cannot be greater than the number of columns in the dataframe.")
    
    # Create a local random state for this function call
    rng = np.random.default_rng()  # Uses a different random state for each function call
    
    # Randomly select n column indices to drop
    columns_to_drop = rng.choice(df.columns, size=n, replace=False)
    # print(columns_to_drop)

    # Drop the selected columns
    df_reduced = df.drop(columns_to_drop, axis=1)
    
    return df_reduced

def split_data(data, train_ratio, val_ratio, test_ratio):
    np.random.seed(42)  # Set random seed for reproducibility
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    # Shuffle the data
    shuffled_data = np.random.permutation(data)

    # Split the data into train, validation, and test sets
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size+val_size]
    test_data = shuffled_data[train_size+val_size:]

    return train_data, val_data, test_data



# For spotify 2
def spotify2_preprocess():
    train_data = pd.read_csv("../../data/external/spotify-2/train.csv")
    test_data = pd.read_csv("../../data/external/spotify-2/test.csv")
    columns_to_drop = ['Unnamed: 0','explicit', 'track_id','track_name', 'album_name', 'artists', 'time_signature', 'key', 'mode']
    train_data.drop(columns_to_drop, axis=1, inplace=True)
    test_data.drop(columns_to_drop, axis=1, inplace=True)

    # Normalize columns with integer or float values using Gaussian normalization
    numeric_columns = test_data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        mean = test_data[col].mean()
        std = test_data[col].std()
        test_data[col] = (test_data[col] - mean) / std

    numeric_columns = train_data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        mean = train_data[col].mean()
        std = train_data[col].std()
        train_data[col] = (train_data[col] - mean) / std

    train_data.drop_duplicates(inplace=True)
    # test_data.drop_duplicates(inplace=True)

    # print(train_data.head())
    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()

    return train_data, test_data

# train_data, test_data = spotify2_preprocess()
# print(train_data.head())

tot_start = time.time()

df = preprocess()
# print(df.columns.tolist())
# Split the data into train, validation, and test sets
train_data, val_data, test_data = split_data(df.to_numpy(), train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

def predict(train_data, test_data, k=15, distance_metric='manhattan', use_gpu=True):
    genre_predict = KNN_model(train_data, k=k, distance_metric=distance_metric, use_gpu=use_gpu)
    # print(np.shape(train_data))

    batch_size = 100
    start_index = 0
    num_rows = len(test_data)
    y_pred_final = []
    y_final = []
    while start_index < num_rows:
        start_time = time.time()
        end_index = min(start_index + batch_size, num_rows)
        # print(f"start: {start_index} end: {end_index}")
        test = test_data[start_index:end_index]
        y_pred, y = genre_predict.inference(test)
        y_pred_final.extend(y_pred)
        y_final.extend(y)
        start_index = end_index
        end_time = time.time()
        print(f"epoch {start_index / batch_size}  {end_time - start_time}")

    performance = performanceMetrics(y_pred_final, y_final)
# performance.printMetrics()
    print("Accuracy: ", performance.accuracy())
    end_time = time.time()
    print((end_time-tot_start))

    return y_pred_final, y_final, performance


def randomDropTests(n, m):
    for i in range(1, n+1):
        for j in range(m+1):
            df = pd.read_csv("../../data/interim/spotify_preprocessed_gaussian.csv")
            df = drop_random_columns(df, i)
            # print(df.columns.tolist())
            # Split the data into train, validation, and test sets
            train_data, val_data, test_data = split_data(df.to_numpy(), train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
            predict(train_data, test_data)


# randomDropTests(6, 5)
# predict(train_data, test_data)