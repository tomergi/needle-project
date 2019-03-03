from sklearn.ensemble import RandomForestClassifier


class Classifier(object):
    """base class for classifiers"""

    def __init__(self):
        pass

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self, X):
        # X can be a list
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError





class RandomForest(Classifier):
    """The random forest classifier"""

    def __init__(self, **kwargs):
        super(RandomForest, self).__init__()
        self.clf = RandomForestClassifier(**kwargs)

    def train(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, X):
        y_hats = self.clf.predict(X)
        return y_hats
