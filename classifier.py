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

fail=0
success=1
TRAIN_FRAC=0.65

MAX_BAG_SIZE = 5000

successful_projects = 0

def load_dataset(filename='result_food.json'):
    global successful_projects
    items = []
    pred = []
    with open(filename, 'r') as fd:
        for row in fd.readlines():
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
                items.append(item)
                if item['csv_state'] == 'successful':
                    pred.append(1)
                    successful_projects += 1
                else:
                    pred.append(0)
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
        #X = self.add_titles(X, items)
        #X = self.add_goal(X, items)
        #X = self.add_time_period(X, items)
        #X = self.add_description(X, items)
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

        #x = self.clean_text(X)
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

        #x = self.clean_text(X)
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
        print(rewards.shape)
        X = np.concatenate((X, rewards.T), axis=1)
        return X


    def train(self,X,y):
        vectorizer = CountVectorizer(analyzer="word", \
                                     tokenizer=None, \
                                     preprocessor=None, \
                                     stop_words=None, \
                                     max_features=5000)

        train_data_features = vectorizer.fit_transform(X)
        np.asarray(train_data_features)

        forest = RandomForestClassifier(n_estimators=100)

        forest = forest.fit(train_data_features, y)
        return forest, vectorizer


    bag_of_word = set()

    def title_to_bag(self, title, bag, bag_indexes):
        translator = str.maketrans('','',string.punctuation)
        stop_words_set = set(stop_words.stop_words)
        vector = np.zeros(len(bag), dtype=np.int32)
        line = title.translate(translator)
        words = line.split(" ")
        for word in words:
            word = word.lower()
            if word and word in bag_indexes and word not in stop_words_set:
                vector[bag_indexes[word]] +=1
        return vector


    def create_bag_of_words(self, X):
        translator = str.maketrans('','',string.punctuation)
        stop_words_set = set(stop_words.stop_words)
        word_set = {}
        for line in X:
            line = line.translate(translator)
            words = line.split(" ")
            for word in words:
                word = word.lower()
                if word and word in word_set and word not in stop_words_set:
                    word_set[word] +=1
                else:
                    word_set[word] = 1

        bag = sorted(list(word_set.items()), key=lambda x: x[1],reverse=True)[:MAX_BAG_SIZE]
        bag = [x[0] for x in bag]

        bag_indexes = {}
        for w in range(len(bag)):
            bag_indexes[bag[w]] = w
        return bag, bag_indexes


    def clean_text(self,X):
        translator = str.maketrans('', '', string.punctuation)
        stop_words_set = set(stop_words.stop_words)
        clean_X = []
        for line in X:
            line = line.translate(translator)
            words = line.split(" ")
            line = ''.join([w.lower() for w in words if w.lower() not in stop_words_set])
            clean_X.append(line)
        return clean_X

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


#train
def train(X,Y):
    #clf = svm.SVC()
    clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0)
    clf.fit(X, Y)
    #svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #        decision_function_shape=None, degree=3, gamma='auto', kernel='sigmoid',
    #        max_iter=-1, probability=False, random_state=None, shrinking=True,
    #        tol=0.001, verbose=False)

    return clf
def getLable(clf, example):
    return clf.predict(np.array([example]))
#validation
def loss(X_valid,Y_valid,X_train,Y_train):
    clf = train(X_train, Y_train)
    visualize_classifier(clf)
    sum = 0
    for example in range( len(X_valid)):
        if(getLable(clf,X_valid[example]) !=Y_valid[example]):
            sum+=1

    return sum/len(X_valid)

def visualize_classifier(model):
    # Extract single tree
    estimator = model.estimators_[0]

    from sklearn.tree import export_graphviz
    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot',
                    rounded = True, proportion = False,
                    precision = 2, filled = True)

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

if __name__ == "__main__":
    # print("start")
    c = Classifier()
    # X,y = load_dataset()
    # X = c.clean_text(X)
    # #hist = c.create_histogram(X)
    # #hist = sorted(list(hist.items()), key=lambda x: x[1],reverse=True)
    # #print(hist)
    #
    #
    # forest,vectorizer = c.train(X,y)
    #
    # l = ["Yes to Europe, No to Thailand Do Political Considerations Influence Israel's List of Travel Warnings?",
    #                  "Israel poised to build 7,000 homes beyond the Green Line",
    #      "At least 35 killed in airstrike on mosque in northern Syria, first responders say",
    #      "PM wins libel suit over claim his wife kicked him out of car"]
    # l = c.clean_text(l)
    # test = np.array(l)
    # test_data_features = vectorizer.transform(test)
    # print(forest.predict(test_data_features))
    X,Y=c.create_features_and_tags()
    (X_train,Y_train),(X_valid,Y_valid) = c.split_train_test(X, Y)
    print('loss rate', loss(X_valid,Y_valid,X_train,Y_train))
    print('sucessful projects percentage %f' % (successful_projects / len(X)))

