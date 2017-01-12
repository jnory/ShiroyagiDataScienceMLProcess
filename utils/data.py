# -*- coding: utf-8 -*-
import os.path
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler


def add_columns(df):
    """
    データフレームの販売価格から査定価格をセットします。
    使用年数も計算します。

    :param df: pandas.DataFrame
    :return: None
    """
    df["appraisal"] = (df["price"] / 2).values.astype(dtype=np.uint64)
    df["years"] = df["dateCreated"].map(lambda x: pd.to_datetime(x)).map(
        lambda x: x.year + x.month / 12.0) - (df["yearOfRegistration"] + df["monthOfRegistration"] / 12.0)


def load(path):
    """
    pathに指定されたCSVファイルを読み込みます。
    pathはautos.csv/train.csv/dev.csv/test.csvへのパスであることが期待されています。

    :param path: CSVへのパス
    :return: pandas.DataFrame
    """
    assert os.path.basename(path) in ("autos.csv", "train.csv", "dev.csv", "test.csv")
    df = pd.read_csv(path, encoding="latin-1")
    if os.path.basename(path) == "autos.csv":
        add_columns(df)
    return df


def split(df, ratio=0.9, seed=131):
    """
    データセットを学習用データ、チューニング用データ、テスト用データに分離します。
    :param df: pandas.DataFrame
    :param ratio: 分離割合
    :param seed: 乱数のシード
    :return: 学習用DataFrame, チューニング用DataFrame, テスト用DataFrame
    """
    train_dev_df, test_df = train_test_split(df, train_size=ratio, random_state=seed)
    train_df, dev_df = train_test_split(train_dev_df, train_size=ratio, random_state=seed+1)
    return train_df, dev_df, test_df


def scale_df(train_df, dev_df, test_df, keys=None):
    """
    特徴量となる値を取り出し、ベクトルを作ります。
    この関数はメーカー情報を考慮しません。
    ベクトルは[-1.0, 1.0]の範囲にスケールされます。
    スケール処理はメーカーごとに行うことに注意してください。

    :param train_df: 学習用DataFrame
    :param dev_df: チューニング用DataFrame
    :param test_df: テスト用DataFrame
    :param keys: 特徴量とする列の名前
    :return:
    学習用特徴量ベクトル, チューニング用特徴量ベクトル, テスト用特徴量ベクトル
    それぞれ、(データ数, 特徴量の次元数)のnumpy配列。
    すなわち、
    [[ 特徴量1 特徴量2 ... 特徴量n], # データフレーム1行目に対応する特徴量
     [ 特徴量1 特徴量2 ... 特徴量n], # データフレーム2行目に対応する特徴量
     ...
     [ 特徴量1 特徴量2 ... 特徴量n]] # データフレームの最後の行に対応する特徴量
     である。特徴量の次元数nはlen(keys)と一致する。
     メーカー情報は含まれない。
    """
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
    """
    特徴量ベクトルの値をメーカー情報も考慮したベクトルに変形します。

    :param brands: 各行の表す車を製造したメーカーの配列, vecと行単位で対応すること。
    [ メーカー1 メーカー2 メーカー1 ... メーカー2 ]

    :param vec: 特徴量ベクトル。(データ数, 特徴量の次元数)
    [[ 特徴量1 特徴量2 ... 特徴量n], # データフレーム1行目に対応する特徴量
     [ 特徴量1 特徴量2 ... 特徴量n], # データフレーム2行目に対応する特徴量
     ...
     [ 特徴量1 特徴量2 ... 特徴量n]] # データフレームの最後の行に対応する特徴量

    :param idx: メーカー名から配置位置を引くための辞書(dict)
    メーカーがm社あるとすると、
    {
        メーカー1: 0,
        メーカー2: 1,
        メーカー3: 2,
        ...
        メーカーm: m-1,
    }
    となるような辞書。

    :return: (データー数, m * (n + 1))となる疎行列(csr_matrix)。
    ある行に記載された情報がメーカーi製の車であるとすると、
      idx[メーカーi] * (n + 1) 列目から idx[メーカーi] * (n + 1) + (n - 1) 列目までに対応するvecの値、
      idx[メーカーi] * (n + 1) + n 列目には1
    を格納する。それ以外の場所は0とする。
    """
    n_features_per_brand = vec.shape[1] + 1
    features = lil_matrix((len(brands), len(idx) * n_features_per_brand))
    for i, (brand, v) in enumerate(zip(brands, vec)):
        j = idx[brand]
        for k in range(v.shape[0]):
            features[i, j * n_features_per_brand + k] = v[k]
        features[i, j * n_features_per_brand + v.shape[0]] = 1
    return features.tocsr()


def convert_to_poly(train_vec, dev_vec, test_vec, degree):
    """
    特徴量ベクトルを多項式版特徴量ベクトルに変換する。

    :param train_vec: 学習用特徴量ベクトル
    :param dev_vec: チューニング用特徴量ベクトル
    :param test_vec: テスト用特徴量ベクトル
    :param degree: 多項式の次数
    :return: 変換した多項式版特徴量ベクトル
    """
    # TODO: 演習1: ここにsklearn.preprocess.PolynomialFeaturesを使ってfeatureを変換するコードを書いてください。
    # TODO: include_biasオプションはFalseにする必要があることに注意してください。
    # 以下のコードはdegree == 1のときのみ有効
    assert degree == 1
    return train_vec, dev_vec, test_vec


def df_to_feature(train_df, dev_df, test_df, degree=1, keys=None):
    """
    データフレームからメーカー情報も考慮した特徴量ベクトルを作ります。

    :param train_df: 学習用DataFrame
    :param dev_df: チューニング用DataFrame
    :param test_df: テスト用DataFrame
    :param degree: モデルの次数
    :param keys: 特徴量に使用する列の名前(list)
    :return: 学習用特徴量ベクトル, チューニング用特徴量ベクトル, テスト用特徴量ベクトル
    いずれもメーカー情報を考慮したベクトル。
    """

    # メーカー情報を考慮しない特徴量ベクトルを作る
    train_vec, dev_vec, test_vec = scale_df(train_df, dev_df, test_df, keys)
    # 多項式版特徴量に変換 (degree==1のときは何もしない)
    train_vec, dev_vec, test_vec = convert_to_poly(train_vec, dev_vec, test_vec, degree)

    # メーカー情報を考慮した特徴量ベクトルに変換
    ubrands = list(set(train_df["brand"].values.tolist()))
    idx = dict(zip(ubrands, range(len(ubrands))))
    train_features = vec_to_feature(train_df["brand"], train_vec, idx)
    dev_features = vec_to_feature(dev_df["brand"], dev_vec, idx)
    test_features = vec_to_feature(test_df["brand"], test_vec, idx)

    return train_features, dev_features, test_features