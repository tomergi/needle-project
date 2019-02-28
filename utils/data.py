import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils import stop_words
import numpy as np
from nltk.stem import porter
from collections import OrderedDict
import utils.deterministic_bag_of_words as deterministic_bag_of_words
from enum import Enum
import re

class IdxToFeature(object):

    def __init__(self, start_idx, end_idx, get_feature_name_func):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.func = get_feature_name_func

    def is_in_range(self, i):
        return self.start_idx <= i <= self.end_idx

    def get_feature(self, i):
        return self.func(i-self.start_idx)

class FeatureExtractor:
    def __init__(self):
        self.IDX_TO_FEATURE_OBJS = []
        self.DESCRIPTION_VECTORIZER = None
        self.title_vectorizer = None
        self.description_vectorizer = None
        self.about_vectorizer = None
    def _add_idx_class(self, X_shape, features_shape, func):
        start = X_shape[1]
        end = (features_shape[1]+start)-1
        o = IdxToFeature(start, end, func)
        self.IDX_TO_FEATURE_OBJS.append(o)


    def get_feature_str_from_col_idx(self, idx):
        for o in self.IDX_TO_FEATURE_OBJS:
            if o.is_in_range(idx):
                return o.get_feature(idx)
        raise ValueError("Could not find the specified idx. this is a bug obviously.")

    
    def _add_bag_of_words(self, X, items):
        bag_of_words = deterministic_bag_of_words.DBoW
        categories_enum = Enum('categories', deterministic_bag_of_words.categories)
        matrix_of_words = []
        for i in range(len(items)):
            words_list = [0]*len(categories_enum)
            description = items[i]['description'].lower()
            result = re.sub(r'-', ' ', description)
            result = re.sub(r'[^A-Za-z 1-9]', '', result)
            for key in bag_of_words:
                if key in result:
                    words_list[categories_enum[bag_of_words[key]].value - 1] = 1
            matrix_of_words.append(words_list)
        npmatrix = np.array(matrix_of_words)
        self._add_idx_class(X_shape=X.shape, features_shape=npmatrix.shape, func=lambda i: "DBoW(%s)" % deterministic_bag_of_words.categories[i])
        X = np.concatenate((X, npmatrix), axis=1)
        return X

    def _add_hour_duration(self, X, items):
        periods_array = np.array([np.array([float(x['hours_duration']) for x in items]).T]).T
        self._add_idx_class(X_shape=X.shape, features_shape=periods_array.shape, func=lambda _: "Hours_Duration")
        return np.concatenate((X, periods_array), axis=1)


    def _add_goal(self, X, items):
        goals = np.array([[float(i['goal']) for i in items]]).T
        self._add_idx_class(X_shape=X.shape, features_shape=goals.shape, func=lambda _: "Goal")
        return np.concatenate((X, goals), axis=1)

    def _add_goal_pledged_ratio(self, X, items):
        pledged = np.array([[float(i['num_pledged']) for i in items]]).T
        goals = np.array([[float(i['goal']) for i in items]]).T
        ratio = pledged/goals
        self._add_idx_class(X_shape=X.shape, features_shape=ratio.shape, func=lambda _: "Pledged_Goal_Ratio")
        return np.concatenate((X, ratio), axis=1)

    def _add_reward_num(self, X, items):
        rewards = np.array([[len(x['rewards']) for x in items]]).T
        self._add_idx_class(X_shape=X.shape, features_shape=rewards.shape, func=lambda _: "NumOfRewards")
        return np.concatenate((X, rewards), axis=1)

    def _add_num_updates(self, X, items):
        num_updates = np.array([[int(i['num_updates']) for i in items]]).T
        self._add_idx_class(X_shape=X.shape, features_shape=num_updates.shape, func=lambda _: "NumOfUpdates")
        return np.concatenate((X, num_updates), axis=1)

    def _add_num_comments(self, X, items):
        num_comments = np.array([[int(i['num_comments']) for i in items]]).T
        self._add_idx_class(X_shape=X.shape, features_shape=num_comments.shape, func=lambda _: "NumOfComments")
        return np.concatenate((X, num_comments), axis=1)

    def _add_num_backers(self, X, items):
        num_backers = np.array([[int(i['num_backers']) for i in items]]).T
        self._add_idx_class(X_shape=X.shape, features_shape=num_backers.shape, func=lambda _: "NumOfBackers")
        return np.concatenate((X, num_backers), axis=1)

    def _add_titles(self, X, items):
        titles = [i['title'] for i in items]
        if not self.title_vectorizer:
            stemmer = porter.PorterStemmer()
            self.title_vectorizer = CountVectorizer(analyzer="word",
                                         preprocessor=stemmer.stem,
                                         stop_words=stop_words.stop_words,
                                         max_features=500)

            features = self.title_vectorizer.fit_transform(titles)
        else:
            features = self.title_vectorizer.transform(titles)
        self._add_idx_class(X_shape=X.shape, features_shape=features.shape, func=lambda i: "Title(%s)" % ([k for k,v in self.title_vectorizer.vocabulary_.items() if v == i][0]))
        return np.concatenate((X, features.toarray()), axis=1)

    def _add_description(self, X, items):
        descriptions = [x['description'] for x in items]
        if not self.description_vectorizer:
            stemmer = porter.PorterStemmer()
            self.description_vectorizer = CountVectorizer(analyzer="word",
                                         preprocessor=stemmer.stem,
                                         stop_words=stop_words.stop_words,
                                         max_features=1000)
            features = self.description_vectorizer.fit_transform(descriptions)
        else:
            features = self.description_vectorizer.transform(descriptions)
        self._add_idx_class(X_shape=X.shape, features_shape=features.shape,
                       func=lambda i: "Description(%s)" % ([k for k, v in self.description_vectorizer.vocabulary_.items() if v == i][0]))
        return np.concatenate((X, features.toarray()), axis=1)


    def _add_about(self, X, items):
        abouts = [x['about'] for x in items]
        if not self.about_vectorizer:
            stemmer = porter.PorterStemmer()
            self.about_vectorizer = TfidfVectorizer(analyzer="word",
                                         preprocessor=stemmer.stem,
                                         stop_words=stop_words.stop_words,
                                         max_features=1000)
            features = self.about_vectorizer.fit_transform(abouts)
        else:
            features = self.about_vectorizer.transform(abouts)
        self._add_idx_class(X_shape=X.shape, features_shape=features.shape, func=lambda i: "About(%s)" % ([k for k,v in self.about_vectorizer.vocabulary_.items() if v == i][0]))
        return np.concatenate((X, features.toarray()), axis=1)


    def extract_features(self, items):
        print("Preprocessing data")
        X = np.empty([len(items), 0])
        print("Preprocessing goal")
        X = self._add_goal(X, items)
        #print("Preprocessing goal_pledged_ratio")
        #X = self._add_goal_pledged_ratio(X, items)
        print("Preprocessing hour_duration")
        X = self._add_hour_duration(X, items)
        print("Preprocessing num_rewards")
        X = self._add_reward_num(X, items)
        print("Preprocessing num_updates")
        X = self._add_num_updates(X, items)
        print("Preprocessing num_comments")
        X = self._add_num_comments(X, items)
        print("Preprocessing titles")
        X = self._add_titles(X, items)
        print("Preprocessing description")
        X = self._add_description(X, items)
        print("Preprocessing about")
        X = self._add_about(X, items)
        print("Preprocessing bag of words")
        X = self._add_bag_of_words(X, items)
        return X


def _modify_json(x, remove_keys):
    x.pop('url')
    if remove_keys is not None:
        for k in remove_keys:
            x.pop(k)

def load_dataset(file_path, max_items=-1, remove_keys=None, shuffle=True, train_split_ratio=0.85):
    if remove_keys is None:
        remove_keys = set()
    else:
        remove_keys = set(remove_keys)
    remove_keys.union({
        'url', 'timeleft', 'creator'
    })
    items = []
    preds = np.array([], dtype=np.uint8)
    print("Starting to load data at: '%s'" % file_path)
    with open(file_path, 'r') as fd:
        for line in fd:
            try:
                if len(items) == max_items:
                    break
                line = line.strip()
                x = json.loads(line[:-1])
                _modify_json(x, remove_keys=remove_keys)  # inplace modifying
                y_str = x.pop('state')
                if y_str == "successful":
                    preds = np.append(preds, 1)
                elif y_str == "failed":  # "failed"
                    preds = np.append(preds, 0)
                else:
                    continue
                items.append(x)
            except:
                continue
    print("Finished loading data.")
    extractor = FeatureExtractor()
    X = extractor.extract_features(items)
    if shuffle is True:
        print("Shuffling data")
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        preds = preds[indices]
    print("Splitting data:")
    split_idx = int(X.shape[0]*train_split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = preds[:split_idx], preds[split_idx:]
    print("Train Data length: %d" % X_train.shape[0])
    print("Test Data length: %d" % X_test.shape[0])
    return extractor, (X_train, Y_train), (X_test, Y_test)



# (X, _), (_, _) = load_dataset("/home/ben/Documents/GitHub/needle-project/dataset/games.json")
# print('---------------')
# print('---------------')
# print('X length: ', X.shape[1])
# print(get_feature_str_from_col_idx(0))
# print(get_feature_str_from_col_idx(5))
# print(get_feature_str_from_col_idx(6))
# print(get_feature_str_from_col_idx(4999))
# print(get_feature_str_from_col_idx(5000))
# print(get_feature_str_from_col_idx(5005))
# print(get_feature_str_from_col_idx(X.shape[1]-1))
# from time import sleep
# sleep(1)
# print(get_feature_str_from_col_idx(X.shape[1]))
# for i in range(X.shape[1]):
#     print(i, get_feature_str_from_col_idx(i))



