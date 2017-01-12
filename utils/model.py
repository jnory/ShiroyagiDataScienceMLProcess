from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


def train(features, price, kind="linear"):
    """
    モデルを学習します。

    :param features: 特徴量ベクトル。priceと行単位で対応している必要があります。
    :param price: 教師データ(査定価格情報)。featuresと行単位で対応している必要があります。
    :param kind: "linear", "lasso", "ridge"のいずれか
    :return: 学習したモデル, 学習データに対する予測値
    """
    # 演習2: 解答例
    if kind == "linear":
        model = LinearRegression(fit_intercept=False)
    elif kind == "lasso":
        model = Lasso(fit_intercept=False)
    elif kind == "ridge":
        model = Ridge(fit_intercept=False)
    else:
        raise ValueError

    model.fit(features, price)
    train_est = model.predict(features)
    return model, train_est


def test(model, features):
    """
    モデルを未知のデータに適用します。

    :param model: train関数で学習したモデル。
    :param features: 特徴量ベクトル。
    :return: 予測結果
    """
    return model.predict(features)
