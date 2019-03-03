import argparse
from utils import scraper, data
import classifiers
import numpy as np
import dill
from subprocess import call
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt


def pred_to_str(y):
    if y == 1:
        return "success"
    else:
        return "failure"


def train_model(max_items=-1):
    extractor, (X_train, Y_train), (X_test, Y_test) = data.load_dataset("games.json", max_items=max_items)
    classifier = classifiers.RandomForest()
    classifier.train(X_train, Y_train)
    test_model(classifier, X_test, Y_test)
    return extractor, classifier

def test_model(classifier, X_test, Y_test):
    predictions = classifier.predict(X_test)
    correct = np.count_nonzero(predictions == Y_test)
    wrong = np.count_nonzero(predictions != Y_test)
    print("success rate: %f, loss rate: %f" % (correct / predictions.size, wrong / predictions.size))
    print("failed projects rate in test set: %f" % (np.count_nonzero(Y_test == 0) / Y_test.size))


def predict(url, extractor, classifier):
    html = scraper.load_page(url)
    item = scraper.parse_page(html)
    item['url'] = url
    items = [item]
    X = extractor.extract_features(items)
    y_hat = classifier.predict(X)[0]
    print("According to the classifier the project result will be %s" % pred_to_str(y_hat))


def visualize_classifier(classifier, extractor):
    # Extract single tree
    model = classifier.clf
    for i in range(len(model.estimators_)):
        title = "word cloud " + str(i)
        estimator = model.estimators_[i]
        features = estimator.feature_importances_
        text_dict = {}
        text_list = []
        for j in range(len(features)):
            name_of_feature = extractor.get_feature_str_from_col_idx(j)
            if features[j] > 0:
                magnitude_of_feature = features[j]
                text_dict[name_of_feature] = magnitude_of_feature
            text_list.append(name_of_feature)
        word_cloud = WordCloud(background_color="white").generate_from_frequencies(text_dict)
        # word_cloud.to_file(title + ".png")
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(title + ".jpg", format="jpg", dpi=1080)
        plt.close()

        title = "tree " + str(i)
        from sklearn.tree import export_graphviz
        # Export as dot file
        export_graphviz(estimator, out_file=title + '.dot', rounded = True, proportion = False, precision = 2, filled = True, class_names=["fail", "success"], feature_names=text_list)

        # Convert to png using system command (requires Graphviz)
        call(['dot', '-Tjpg', title + '.dot', '-o', title + '.jpg', '-Gdpi=1080'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="file containing the model and feature extractor")
    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True
    # Training Parser:
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument("--visualize", "-v", help="visualize the classifier", action="store_true")
    train_parser.add_argument("--max_items", "-m", help="max number of projects to train and test on", type=int, default=-1)
    # Predicting Parser:
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument('url', help="The url to the webpage of the Kickstarter project")
    args = parser.parse_args()
    if args.mode == "train":
        extractor, classifier = train_model(args.max_items)
        with open(args.model, "wb") as model:
            dill.dump((extractor, classifier), model)
        if args.visualize:
            visualize_classifier(classifier, extractor)
    else:
        with open(args.model, "rb") as model:
            extractor, classifier = dill.load(model)
        predict(args.url, extractor, classifier)


if __name__ == '__main__':
    main()
