class DumbModel:
    """Dumb model always predict 0"""
    def fit(self, X, y):
        pass

    def predict(self, X):
        return [0] * len(X)

    def dump(self, filename_output):
        pass
