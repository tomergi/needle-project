import argparse
from utils import scraper
import classifiers
import dill
import copy

def predict(item, extractor, classifier):
    """returns the chances of success for a project"""
    items = [item]
    X = extractor.extract_features(items)
    return classifier.clf.predict_proba(X)[0][1]

def suggest_improvements(url, extractor, classifier):
    """ Suggest improvements to the given project. The changes to the project that increase the success probability are printed at the end"""
    html = scraper.load_page(url)
    item = scraper.parse_page(html)

    best_probability = predict(item, extractor, classifier)
    base_probability = best_probability
    best_item = item

    # do two rounds of optimizing the item, changing one parameter at a time
    for i in range(2):
        for updates_num in range(10):
            new_item = copy.deepcopy(best_item)
            new_item['num_updates'] = updates_num
            probability = predict(new_item, extractor, classifier)
            if probability > best_probability:
                best_probability = probability
                best_item = new_item
        for num_comments in range(0, 2000, 20):
            new_item = copy.deepcopy(best_item)
            new_item['num_comments'] = num_comments
            probability = predict(new_item, extractor, classifier)
            if probability > best_probability:
                best_probability = probability
                best_item = new_item
        for rewards_num in range(20):
            new_item = copy.deepcopy(best_item)
            new_item['rewards'] = [0] * rewards_num
            probability = predict(new_item, extractor, classifier)
            if probability > best_probability:
                best_probability = probability
                best_item = new_item
        #test for goals in the range of 0.5-1.5 of the current goal
        for goal_multiplier in range(5, 16):
            new_item = copy.deepcopy(best_item)
            new_item['goal'] = item['goal'] * goal_multiplier / 10
            probability = predict(new_item, extractor, classifier)
            if probability > best_probability:
                best_probability = probability
                best_item = new_item
        #projects can last from one day to 60 days
        for duration in range(24, 24 * 60):
            new_item = copy.deepcopy(best_item)
            new_item['hours_duration'] = duration
            probability = predict(new_item, extractor, classifier)
            if probability > best_probability:
                best_probability = probability
                best_item = new_item
    print("Initial success probability: %f, best success probability: %f" % (base_probability, best_probability))
    if best_probability > base_probability:
        print("To get the best probability of success, you should:")
        if len(best_item['rewards']) != len(item['rewards']):
            print(" Have %d rewards" % len(best_item['rewards']))
        if best_item['num_comments'] != item['num_comments']:
            print(" Get %d people to comment" % best_item['num_comments'])
        if best_item['goal'] != item['goal']:
            print(" Change the goal to %f" % best_item['goal'])
        if best_item['hours_duration'] != item['hours_duration']:
            print(" Change project duration to be %d hours" % best_item['hours_duration'])
        if best_item['num_updates'] != item['num_updates']:
            print(" Have %d updates in your project" % new_item['num_updates'])
    else:
        print("Could not suggest improvements for the project")

def main():
    parser = argparse.ArgumentParser("Give suggestions to a kickstarter project to improve success chances")
    parser.add_argument("model", help="model file (output of main.py training)")
    parser.add_argument("url", help="project url")
    args = parser.parse_args()

    with open(args.model, "rb") as model:
        extractor, classifier = dill.load(model)
    suggest_improvements(args.url, extractor, classifier)

if __name__ == "__main__":
    main()
