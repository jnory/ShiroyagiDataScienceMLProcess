import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load(path):
    df = pd.read_csv(path, encoding="latin-1")
    df["appraisal"] = (df["price"] / 2).values.astype(dtype=np.uint64)
    df["years"] = df["dateCreated"].map(lambda x: pd.to_datetime(x)).map(
        lambda x: x.year + x.month / 12.0) - (df["yearOfRegistration"] + df["monthOfRegistration"] / 12.0)
    return df


def split(df, ratio=0.9, seed=131):
    train_dev_df, test_df = train_test_split(df, train_size=ratio, random_state=seed)
    train_df, dev_df = train_test_split(train_dev_df, train_size=ratio, random_state=seed+1)
    return train_df, dev_df, test_df


def standardize_df(train_df, dev_df, test_df, keys=None):
    if keys is None:
        keys = ["years"]

    brands = set(train_df["brand"].values.tolist())
    brands_dev = set(dev_df["brand"].values.tolist())
    brands_test = set(test_df["brand"].values.tolist())
    assert brands_dev.issubset(brands)
    assert brands_test.issubset(brands)

    train_vec = train_df[keys].values.astype(np.float).copy()
    dev_vec = dev_df[keys].values.astype(np.float).copy()
    test_vec = test_df[keys].values.astype(np.float).copy()
    for b in brands:
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        train_vec[(train_df["brand"] == b).values] = scaler.fit_transform(
            train_df[train_df["brand"] == b][keys].values.astype(np.float))
        dev_vec[(dev_df["brand"] == b).values] = scaler.transform(
            dev_df[dev_df["brand"] == b][keys].values.astype(np.float))
        test_vec[(test_df["brand"] == b).values] = scaler.transform(
            test_df[test_df["brand"] == b][keys].values.astype(np.float))
    return train_vec, dev_vec, test_vec


def vec_to_feature(brands, vec, idx):
    features = lil_matrix((len(brands), len(idx) * (vec.shape[1] + 1)))
    for i, (brand, v) in enumerate(zip(brands, vec)):
        j = idx[brand]
        for k in range(v.shape[0]):
            features[i, j * (v.shape[0] + 1) + k] = v[k]
        features[i, j * (v.shape[0] + 1) + v.shape[0]] = 1
    return features.tocsr()


def convert_to_poly(train_vec, dev_vec, test_vec, degree):
    # TODO: 演習1: ここにsklearn.preprocess.PolynomialFeaturesを使ってfeatureを変換するコードを書いてください。
    # 以下のコードはdegree == 1のときのみ有効
    assert degree == 1
    return train_vec, dev_vec, test_vec


def df_to_feature(train_df, dev_df, test_df, degree=1):
    train_vec, dev_vec, test_vec = standardize_df(train_df, dev_df, test_df)
    train_vec, dev_vec, test_vec = convert_to_poly(train_vec, dev_vec, test_vec, degree)

    ubrands = list(set(train_df["brand"].values.tolist()))
    idx = dict(zip(ubrands, range(len(ubrands))))

    train_features = vec_to_feature(train_df["brand"], train_vec, idx)
    dev_features = vec_to_feature(dev_df["brand"], dev_vec, idx)
    test_features = vec_to_feature(test_df["brand"], test_vec, idx)

    return train_features, dev_features, test_features