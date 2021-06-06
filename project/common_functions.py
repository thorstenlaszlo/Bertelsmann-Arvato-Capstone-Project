import datetime
from time import time

import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def is_numeric(val):
    try:
        float(val)
        return True
    except Exception:
        return False


def dump_distinct_non_null_values(df):
    """
    Print distinct non null Elements of each column of a DataFrame
    :param df: The DataFrame
    :return: None
    """
    for col in df.columns:
        values = sorted([elem for elem in df[col].unique() if not pd.isnull(elem)], key=lambda it: str(it))
        print(f'{col}, ({len(values)}) distinct elements, content:')
        print(values, '\n')


def load_dataframe(filename, index_col='LNR'):
    """Read HDF-File into DataFrame"""
    df = pd.read_hdf(filename, key='df')
    df = df.set_index(index_col)
    return df


def save_dataframe(df, filename):
    df.reset_index().to_hdf(filename, key='df', mode='w', index=False)


def load_model(filename):
    with open(filename, 'rb') as rf:
        model = joblib.load(rf)
    return model


def save_model(model, filename):
    with open(filename, 'wb') as wf:
        joblib.dump(model, wf)


def numpy_array_to_df(np_data):
    """
    Convert a numpy array to a pandas DataFrame.
    :param np_data: the numpy array containing the data
    :return: A pandas DataFrame
    """
    cols = [f"col_{i}" for i in range(np_data.shape[1])]
    return pd.DataFrame(columns=cols, data=np_data)


# adapted from https://stackoverflow.com/questions/64967847/pandas-representative-sampling-across-multiple-columns
def representative_sample(df, sample_size):
    """
    Create a representative sample of data frame by creating a combined feature,
    weight it and draw subsample with it as weight.
    :param df: The DataFame
    :param sample_size: size of the sample
    :return: the DataFrame containing the sample
    """
    df_ = df.copy(deep=True)
    df_['combined'] = list(zip(*[df_[col] for col in df_.columns]))
    combined_weight = df_['combined'].value_counts(normalize=True).to_dict()
    df_['combined_weight'] = df_['combined'].apply(lambda x: combined_weight[x])
    df_sample = df_.sample(sample_size, weights=df_['combined_weight'])
    return df_sample.drop(columns=['combined', 'combined_weight'], axis=1)


def fitted_k_mean_models(points, num_clusters, max_iter=100, verbose=1):
    """
    Generate a list of tuples (k, fitted_K-Means_model_for_k)
    :param points: The data to fit the K-Means estimator against
    :param num_clusters: the n_clusters of the underlying KMeans estimator
    :param verbose: the verbosity
    :return: An array of tuples (k, fitted_K-Means_model_for_k)
    """
    kmeans = []
    for k in range(num_clusters[0], num_clusters[1] + 1):
        if verbose:
            t = time()
            print(f"Calculate Kmeans for {k} clusters.")
        kmeans.append((k, KMeans(n_clusters=k, max_iter=max_iter, algorithm='elkan').fit(points)))
        if verbose:
            delta = datetime.timedelta(seconds=(time() - t))
            print("Done in:", delta)
    return kmeans


def kmeans_scores(points, kmeans_array, verbose=1):
    """
    Generate a list of K-Means scores (SSE) by applying a list
    of KMeans estimators on the input data and calling their score method
    :param points: The data
    :param kmeans_array: list tuples containing the estimators - can be generated with fitted_k_mean_models(...) above
    :param verbose: the verbosity
    :return: list of the K-Means scores
    """
    scores = []
    for item in kmeans_array:
        k = item[0]
        kmeans = item[1]
        if verbose:
            t = time()
            print(f"Calculate kmeans_scores (sse) for {k} clusters.")
        sse = abs(kmeans.score(points))
        scores.append((k, sse))
        if verbose:
            delta = datetime.timedelta(seconds=(time() - t))
            print("Score", sse, "Done in:", delta)
    return scores


def silhouette_scores(points, kmeans_array, sample_size=40000, verbose=1):
    """
    Generate a list of silhouette scores by applying the silhouette_score function on a list
    of KMeans estimators and the input data and calling their score method
    :param points: The data
    :param kmeans_array: list tuples containing the estimators - can be generated with fitted_k_mean_models(...) above
    :param sample_size: the sample_size parameter of the underlying silhouette_score method
    :param verbose: the verbosity
    :return: list of the K-Means scores
    """
    scores = []
    for item in kmeans_array:
        k = item[0]
        kmeans = item[1]
        if verbose:
            t = time()
            print(f"Calculate silhouette_scores for {k} clusters.")
        sil = silhouette_score(points, kmeans.predict(points), metric='euclidean', sample_size=sample_size)
        scores.append((k, sil))
        if verbose:
            delta = datetime.timedelta(seconds=(time() - t))
            print("Score", sil, ", Done in:", delta)
    return scores
