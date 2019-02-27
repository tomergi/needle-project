import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils import stop_words
import numpy as np
from nltk.stem import porter
from collections import OrderedDict


DESCRIPTION_VECTORIZER = None
IDX_TO_FEATURE_OBJS = []

class IdxToFeature(object):

    def __init__(self, start_idx, end_idx, get_feature_name_func):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.func = get_feature_name_func
        print(start_idx, end_idx)

    def is_in_range(self, i):
        return self.start_idx <= i <= self.end_idx

    def get_feature(self, i):
        return self.func(i-self.start_idx)


def _add_idx_class(X_shape, features_shape, func):
    start = X_shape[1]
    end = (features_shape[1]+start)-1
    o = IdxToFeature(start, end, func)
    IDX_TO_FEATURE_OBJS.append(o)


def get_feature_str_from_col_idx(idx):
    for o in IDX_TO_FEATURE_OBJS:
        if o.is_in_range(idx):
            return o.get_feature(idx)
    raise ValueError("Could not find the specified idx. this is a bug obviously.")


def _add_hour_duration(X, items):
    periods_array = np.array([np.array([float(x['hours_duration']) for x in items]).T]).T
    _add_idx_class(X_shape=X.shape, features_shape=periods_array.shape, func=lambda _: "Hours_Duration")
    return np.concatenate((X, periods_array), axis=1)


def _add_goal(X, items):
    goals = np.array([[float(i['goal']) for i in items]]).T
    _add_idx_class(X_shape=X.shape, features_shape=goals.shape, func=lambda _: "Goal")
    return np.concatenate((X, goals), axis=1)


def _add_goal_pledged_ratio(X, items):
    pledged = np.array([[float(i['num_pledged']) for i in items]]).T
    goals = np.array([[float(i['goal']) for i in items]]).T
    ratio = pledged/goals
    _add_idx_class(X_shape=X.shape, features_shape=ratio.shape, func=lambda _: "Pledged_Goal_Ratio")
    return np.concatenate((X, ratio), axis=1)

def _add_reward_num(X, items):
    rewards = np.array([[len(x['rewards']) for x in items]]).T
    _add_idx_class(X_shape=X.shape, features_shape=rewards.shape, func=lambda _: "NumOfRewards")
    return np.concatenate((X, rewards), axis=1)

def _add_num_updates(X, items):
    num_updates = np.array([[int(i['num_updates']) for i in items]]).T
    _add_idx_class(X_shape=X.shape, features_shape=num_updates.shape, func=lambda _: "NumOfUpdates")
    return np.concatenate((X, num_updates), axis=1)

def _add_num_comments(X, items):
    num_comments = np.array([[int(i['num_comments']) for i in items]]).T
    _add_idx_class(X_shape=X.shape, features_shape=num_comments.shape, func=lambda _: "NumOfComments")
    return np.concatenate((X, num_comments), axis=1)

def _add_num_backers(X, items):
    num_backers = np.array([[int(i['num_backers']) for i in items]]).T
    _add_idx_class(X_shape=X.shape, features_shape=num_backers.shape, func=lambda _: "NumOfBackers")
    return np.concatenate((X, num_backers), axis=1)

def _add_titles(X, items):
    titles = [i['title'] for i in items]
    stemmer = porter.PorterStemmer()
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=stemmer.stem,
                                 stop_words=stop_words.stop_words,
                                 max_features=5000)

    features = vectorizer.fit_transform(titles)
    _add_idx_class(X_shape=X.shape, features_shape=features.shape, func=lambda i: "Title(%s)" % ([k for k,v in vectorizer.vocabulary_.items() if v == i][0]))
    return np.concatenate((X, features.toarray()), axis=1)

def _add_description(X, items):
    stemmer = porter.PorterStemmer()
    descriptions = [x['description'] for x in items]
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=stemmer.stem,
                                 stop_words=None,
                                 max_features=15000)
    features = vectorizer.fit_transform(descriptions)
    _add_idx_class(X_shape=X.shape, features_shape=features.shape,
                   func=lambda i: "Description(%s)" % ([k for k, v in vectorizer.vocabulary_.items() if v == i][0]))
    return np.concatenate((X, features.toarray()), axis=1)


def _add_about(X, items):
    abouts = [x['about'] for x in items]
    stemmer = porter.PorterStemmer()
    vectorizer = TfidfVectorizer(input='content', encoding='utf-8',
                                 decode_error='strict', strip_accents=None, lowercase=True,
                                 preprocessor=stemmer.stem, tokenizer=None, analyzer='word',
                                 stop_words=stop_words.stop_words, token_pattern=r"(?u)\b\w\w+\b",
                                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                                 max_features=15000, vocabulary=None, binary=False,
                                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                                 sublinear_tf=False)
    features = vectorizer.fit_transform(abouts)
    _add_idx_class(X_shape=X.shape, features_shape=features.shape, func=lambda i: "About(%s)" % ([k for k,v in vectorizer.vocabulary_.items() if v == i][0]))
    return np.concatenate((X, features.toarray()), axis=1)


def _preprocess(items, preds, shuffle, split_ratio):
    assert 0 < split_ratio < 1, "Ratio must in range (0,1)"
    print("Preprocessing data")
    X = np.empty([len(items), 0])
    print("Preprocessing goal")
    X = _add_goal(X, items)
    print("Preprocessing goal_pledged_ratio")
    X = _add_goal_pledged_ratio(X, items)
    print("Preprocessing hour_duration")
    X = _add_hour_duration(X, items)
    print("Preprocessing num_rewards")
    X = _add_reward_num(X, items)
    print("Preprocessing num_updates")
    X = _add_num_updates(X, items)
    print("Preprocessing num_comments")
    X = _add_num_comments(X, items)
    print("Preprocessing titles")
    X = _add_titles(X, items)
    print("Preprocessing description")
    X = _add_description(X, items)
    print("Preprocessing about")
    X = _add_about(X, items)

    if shuffle is True:
        print("Shuffling data")
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        preds = preds[indices]
    print("Splitting data:")
    split_idx = int(X.shape[0]*split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = preds[:split_idx], preds[split_idx:]
    print("Train Data length: %d" % X_train.shape[0])
    print("Test Data length: %d" % X_test.shape[0])
    return (X_train, Y_train), (X_test, Y_test)


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
            if len(items) == max_items:
                break
            line = line.strip()
            x = json.loads(line)
            _modify_json(x, remove_keys=remove_keys)  # inplace modifying
            y_str = x.pop('state')
            if y_str == "successful":
                preds = np.append(preds, 1)
            elif y_str == "failed":  # "failed"
                preds = np.append(preds, 0)
            else:
                continue
            items.append(x)
    print("Finished loading data.")

    (X_train, Y_train), (X_test, Y_test) = _preprocess(items, preds, shuffle, train_split_ratio)
    return (X_train, Y_train), (X_test, Y_test)



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



