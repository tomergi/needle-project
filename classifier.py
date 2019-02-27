from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import string
import stop_words
import numpy as np
from sklearn import svm
from sklearn import linear_model
import json
from nltk.stem import porter
from datetime import datetime
import lxml
import lxml.html
import deterministic_bag_of_words
from enum import Enum
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from PIL import Image

fail=0
success=1
TRAIN_FRAC=0.75

MAX_BAG_SIZE = 5000
MAX_ITEMS = 15000

successful_projects = 0

def load_dataset(filename='result_games.json'):
    global successful_projects
    items = []
    pred = []
    with open(filename, 'r') as fd:
        for row in fd:
            try:
                item = json.loads(row.strip()[:-1])
                _, csv_id, csv_name, csv_category, csv_main_category, csv_currency, csv_deadline, csv_goal, csv_launched, csv_pledged, csv_state, csv_backers, csv_country, csv_usd_pledged, csv_pledged_real, csv_usd_goal_real = item['csv_row']
                item['csv_id'] = csv_id
                item['csv_name'] = csv_name
                item['csv_category'] = csv_category
                item['csv_main_category'] = csv_main_category
                item['csv_currency'] = csv_currency
                item['csv_deadline'] = csv_deadline
                item['csv_goal'] = csv_goal
                item['csv_launched'] = csv_launched
                item['csv_pledged'] = csv_pledged
                item['csv_state'] = csv_state
                item['csv_backers'] = csv_backers
                item['csv_country'] = csv_country
                item['csv_usd_pledged'] = csv_usd_pledged
                item['csv_pledged_real'] = csv_pledged_real
                item['csv_usd_goal_real'] = float(csv_usd_goal_real)
                item['description'] = "bla bla bla, rpg is magic rts is sucks"
                if item['csv_state'] == 'canceled':
                    continue
                items.append(item)
                if item['csv_state'] == 'successful':
                    pred.append(1)
                    successful_projects += 1
                else:
                    pred.append(0)
                if len(items) > MAX_ITEMS:
                    break
            except:
                continue

    return items, np.array(pred)


class Classifier(object):

    #def __init__(self):
        ##self.X, self.y = load_dataset()
        #self.bag_of_word = self.create_bag_of_words(X)


    def classify(self,X):
        pass

    def create_features_and_tags(self):
        items, y = load_dataset()

        X = np.array([]).reshape(len(items), 0)
        X = self.add_titles(X, items)
        X = self.add_goal(X, items)
        X = self.add_time_period(X, items)
        X = self.add_description(X, items)
        X = self.add_reward_num(X, items)
        X = self.add_bag_of_words(X, items)
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
            td /= 3600 # in hours
            periods.append(td)
        periods_array = np.array([periods]).T
        # print(X.shape, periods_array.shape)
        # exit(123)
        X = np.concatenate((X, periods_array), axis=1)
        return X

    def add_bag_of_words(self, X, items):
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
        npmatrix = np.array(matrix_of_words).T
        for column in npmatrix:
            column.shape = (len(column), 1)
            X = np.concatenate((X, column), axis=1)
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
        # print(rewards.shape)
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

def train(X,Y):
    #clf = svm.SVC()
    clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0)
    clf.fit(X, Y)
    #svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #       decision_function_shape=None, degree=3, gamma='auto', kernel='sigmoid',
    #       max_iter=-1, probability=False, random_state=None, shrinking=True,
    #       tol=0.001, verbose=False)

    return clf
def getLable(clf, example):
    return clf.predict(np.array([example]))

#validation
def loss(X_valid,Y_valid,X_train,Y_train):
    print("calc")
    clf = train(X_train, Y_train)
    visualize_classifier(clf)
    sum = 0
    for example in range( len(X_valid)):
        if(getLable(clf,X_valid[example]) !=Y_valid[example]):
            sum+=1

    return sum/len(X_valid)

def magic(number):
    return str(number)

def visualize_classifier(model):
    # Extract single tree
    for i in range(len(model.estimators_)):
        title = "word cloud " + str(i)
        estimator = model.estimators_[i]
        features = estimator.feature_importances_
        text_dict = {}
        text_list = []
        for j in range(len(features)):
            name_of_feature = magic(j)
            if features[j] > 0:
                magnitude_of_feature = features[j]
                text_dict[name_of_feature] = magnitude_of_feature
            text_list.append(name_of_feature)
        word_cloud = WordCloud(background_color="white").generate_from_frequencies(text_dict)
        # word_cloud.to_file(title + ".png")
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(title + ".png", format="PNG", dpi=1080)
        plt.close()

        title = "tree " + str(i)
        from sklearn.tree import export_graphviz
        # Export as dot file
        export_graphviz(estimator, out_file=title + '.dot', rounded = True, proportion = False, precision = 2, filled = True, class_names=["fail", "success"], feature_names=text_list)

        # Convert to png using system command (requires Graphviz)
        from subprocess import call
        call(['dot', '-Tpng', title + '.dot', '-o', title + '.png', '-Gdpi=1080'])

if __name__ == "__main__":
    c = Classifier()
    X,Y=c.create_features_and_tags()
    (X_train,Y_train),(X_valid,Y_valid) = c.split_train_test(X, Y)
    print('loss rate', loss(X_valid,Y_valid,X_train,Y_train))
    print('sucessful projects percentage %f' % (successful_projects / len(X)))

