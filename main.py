import argparse
from utils import scraper, data
import classifiers


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='mode')
subparsers.required = True
# Training Parser:
train_parser = subparsers.add_parser('train')
# Predicting Parser:
predict_parser = subparsers.add_parser("predict")
predict_parser.add_argument('url', help="The url to the webpage of the Kickstarter project")
ARGS = parser.parse_args()


def pred_to_str(y):
    if y == 0:
        return "success"
    else:
        return "failure"


def train_model():
    X, Y = data.load_dataset("./data/games.csv")
    classifiers.RandomForest()


def test_model():
    html = scraper.load_page(ARGS.url)
    x = scraper.parse_page(html)
    classifier_category = str.lower(x['category'])
    if classifier_category == 'games':
        model = load_model("./models/games")
        y_hat = model.predict(x)
    else:
        raise ValueError("Sorry but there isn't a model trained for the category: '%s'" % classifier_category)
    print("According to the classifier the project will %s" % pred_to_str(y_hat))


def main():
    if ARGS.mode == "train":
        train_model()
    else:  # ARGS.mode == "predict"
        test_model()


if __name__ == '__main__':
    main()
