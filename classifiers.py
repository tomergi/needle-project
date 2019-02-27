from sklearn.ensemble import RandomForestClassifier


class Classifier(object):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def preprocess(self):
        # self.X = modify...
        # self.Y = modify...
        raise NotImplementedError

    def predict(self, X):
        # X can be a list
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError





class RandomForest(Classifier):

    def __init__(self,
                 X,
                 Y,
                 n_estimators='warn',
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RandomForest, self).__init__(X=X, Y=Y)
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
        )
        self.preprocess()

    def train(self):
        self.clf.fit(self.X, self.Y)

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
