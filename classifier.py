from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import stop_words
import numpy as np
from nltk.stem import porter
from datetime import datetime
import lxml
import lxml.html
from get_data import load_dataset
fail = 0
success = 1
TRAIN_FRAC = 0.75

MAX_ITEMS = 15000

successful_projects = 0


class Classifier(object):

    # def __init__(self):
    ##self.X, self.y = load_dataset()
    # self.bag_of_word = self.create_bag_of_words(X)

    def classify(self, X):
        pass

    def create_features_and_tags(self):
        items, y = load_dataset("./data/usd_Games_big.json", max_items=MAX_ITEMS)
        X = np.array([]).reshape(len(items), 0)
        # X = self.add_titles(X, items)
        X = self.add_goal(X, items)
        X = self.add_time_period(X, items)
        X = self.add_description(X, items)
        X = self.add_reward_num(X, items)
        return X, np.array(y)

    def add_titles(self, X, items):
        titles = [i['csv_name'] for i in items]
        stemmer = porter.PorterStemmer()
        vectorizer = CountVectorizer(analyzer="word", \
                                     tokenizer=None, \
                                     preprocessor=stemmer.stem, \
                                     stop_words=stop_words.stop_words, \
                                     max_features=50000)

        train_data_features = vectorizer.fit_transform(titles)
        X = np.concatenate((X, train_data_features.toarray()), axis=1)
        return X

    def add_time_period(self, X, items):
        periods = []
        for i in range(len(X)):
            start = datetime.strptime(items[i]['csv_launched'], "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(items[i]['csv_deadline'], "%Y-%m-%d")
            td = (end - start).total_seconds()
            td /= 3600  # in hours
            periods.append(td)
        periods_array = np.array([periods]).T

        X = np.concatenate((X, periods_array), axis=1)
        return X

    def add_description(self, X, items):
        descriptions = []
        for item in items:
            r = lxml.html.fromstring(item['Text'])
            description = r.xpath("/html/head/meta[contains(@name, 'description')]")[0].attrib['content']
            descriptions.append(description)
        stemmer = porter.PorterStemmer()
        vectorizer = CountVectorizer(analyzer="word", \
                                     tokenizer=None, \
                                     preprocessor=stemmer.stem, \
                                     stop_words=None, \
                                     max_features=50000)

        train_data_features = vectorizer.fit_transform(descriptions)
        X = np.concatenate((X, train_data_features.toarray()), axis=1)
        return X

    def add_goal(self, X, items):
        goals = np.array([[i['csv_usd_goal_real'] for i in items]]).T

        X = np.concatenate((X, goals), axis=1)
        return X

    def add_reward_num(self, X, items):
        rewards = []
        for item in items:
            rewards.append(len(item['rewards']))
        rewards = np.array([rewards])
        X = np.concatenate((X, rewards.T), axis=1)
        return X

    def split_train_test(self, X, y):
        indx = np.arange(len(X))
        np.random.shuffle(indx)

        sep = np.int(TRAIN_FRAC * len(X))
        X = np.array(X)
        y = np.array(y)

        X_train = X[indx[:sep]]
        Y_train = y[indx[:sep]]
        X_test = X[indx[sep:]]
        Y_test = y[indx[sep:]]

        return (X_train, Y_train), (X_test, Y_test)


def train(X, Y):
    # clf = svm.SVC()
    clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0)
    clf.fit(X, Y)
    # svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #       decision_function_shape=None, degree=3, gamma='auto', kernel='sigmoid',
    #       max_iter=-1, probability=False, random_state=None, shrinking=True,
    #       tol=0.001, verbose=False)

    return clf


def getLable(clf, example):
    return clf.predict(np.array([example]))


# validation
def loss(X_valid, Y_valid, X_train, Y_train):
    clf = train(X_train, Y_train)
    # visualize_classifier(clf)
    sum = 0
    for example in range(len(X_valid)):
        if (getLable(clf, X_valid[example]) != Y_valid[example]):
            sum += 1

    return sum / len(X_valid)


def visualize_classifier(model):
    # Extract single tree
    estimator = model.estimators_[0]

    from sklearn.tree import export_graphviz
    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot',
                    rounded=True, proportion=False,
                    precision=2, filled=True, class_names=["fail", "success"])

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])


if __name__ == "__main__":
    c = Classifier()
    X, Y = c.create_features_and_tags()
    (X_train, Y_train), (X_valid, Y_valid) = c.split_train_test(X, Y)
    print('loss rate', loss(X_valid, Y_valid, X_train, Y_train))
    print('sucessful projects percentage %f' % (successful_projects / len(X)))
