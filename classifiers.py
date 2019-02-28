from sklearn.ensemble import RandomForestClassifier


class Classifier(object):

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

    def __init__(self, **kwargs):
        super(RandomForest, self).__init__()
        self.clf = RandomForestClassifier(**kwargs)

    def train(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, X):
        y_hats = self.clf.predict(X)
        return y_hats


    # def visualize(self):
    #     # Extract single tree
    #     # estimator = model.estimators_[0]
    #     #
    #     # from sklearn.tree import export_graphviz
    #     # # Export as dot file
    #     # export_graphviz(estimator, out_file='tree.dot',
    #     #                 rounded=True, proportion=False,
    #     #                 precision=2, filled=True, class_names=["fail", "success"])
    #     #
    #     # # Convert to png using system command (requires Graphviz)
    #     # from subprocess import call
    #     # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    #     pass






# # validation
# def loss(X_valid, Y_valid, X_train, Y_train):
#     clf = train(X_train, Y_train)
#     # visualize_classifier(clf)
#     sum = 0
#     for example in range(len(X_valid)):
#         if (getLable(clf, X_valid[example]) != Y_valid[example]):
#             sum += 1
#
#     return sum / len(X_valid)



# if __name__ == "__main__":
#     c = RandomForest()
#     X, Y = c.create_features_and_tags()
#     (X_train, Y_train), (X_valid, Y_valid) = c.split_train_test(X, Y)
#     print('loss rate', loss(X_valid, Y_valid, X_train, Y_train))
#     print('sucessful projects percentage %f' % (successful_projects / len(X)))
