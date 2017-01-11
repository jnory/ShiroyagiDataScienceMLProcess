from sklearn.linear_model import LinearRegression


def train(features, price, kind="linear"):
    # TODO: 演習2 kind == "lasso"のときLasso回帰、 kind == "ridge"のときRidge回帰を実行するように修正してください。
    assert kind == "linear"
    model = LinearRegression(fit_intercept=False)
    model.fit(features, price)
    train_est = model.predict(features)
    return model, train_est


def test(model, features):
    return model.predict(features)