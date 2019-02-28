import argparse
from utils import scraper, data
import classifiers
import numpy as np
import dill


def pred_to_str(y):
    if y == 1:
        return "success"
    else:
        return "failure"


def train_model():
    extractor, (X_train, Y_train), (X_test, Y_test) = data.load_dataset("games.json", max_items=10000)
    classifier = classifiers.RandomForest()
    classifier.train(X_train, Y_train)
    test_model(classifier, X_test, Y_test)
    return extractor, classifier

def test_model(classifier, X_test, Y_test):
    print(X_test)
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
    y_hat = classifier.predict(X)
    print("According to the classifier the project result will be %s" % pred_to_str(y_hat))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="file containing the model and feature extractor")
    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True
    # Training Parser:
    train_parser = subparsers.add_parser('train')
    # Predicting Parser:
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument('url', help="The url to the webpage of the Kickstarter project")
    args = parser.parse_args()
    if args.mode == "train":
        extractor, classifier = train_model()
        with open(args.model, "wb") as model:
            dill.dump((extractor, classifier), model)
    else:
        with open(args.model, "rb") as model:
            extractor, classifier = dill.load(model)
        predict(args.url, extractor, classifier)


if __name__ == '__main__':
    main()
