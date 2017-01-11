from sklearn.linear_model import LinearRegression


def train(features, price, kind="linear"):
    """
    モデルを学習します。

    :param features: 特徴量ベクトル。priceと行単位で対応している必要があります。
    :param price: 教師データ(査定価格情報)。featuresと行単位で対応している必要があります。
    :param kind: "linear", "lasso", "ridge"のいずれか
    :return: 学習したモデル, 学習データに対する予測値
    """
    # TODO: 演習2 kind == "lasso"のときLasso回帰、 kind == "ridge"のときRidge回帰を実行するように修正してください。
    # TODO: sklearn.linear_model.Lasso, sklearn.linear_model.Ridgeを使います。
    assert kind == "linear"
    model = LinearRegression(fit_intercept=False)
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