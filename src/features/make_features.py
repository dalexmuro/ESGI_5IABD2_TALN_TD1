def make_features(df):
    y = df["is_comic"]
    X = df.drop(columns=["is_comic"])

    return X, y
