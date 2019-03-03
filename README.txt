Code for our project.

The pipeline of the project looks somewhat like so:
1. Extract the games category from the ks-projects-201801.csv using utils/csv_creator.py
2. Scrape them using kickstarter_scrape.py
3. Convert them to a better format using scraper.py
4. Train the classifier on them using main.py
5. Use the trained model to suggest improvements to the project using kickstarter_doctor.py

Files:
main.py - trains and tests the classifier, can be used to predict the success/failure of a single project
kickstarter_doctor.py - suggests improvements for a running project to increase its success chance
classifiers.py - the random forest classifier we used
utils/data.py - feature extraction from the scraped and parsed project
utils/kickstarter_scrape.py - a scraper using selenium that can run over a list of pages and extract their html and some other useful data
utils/scraper.py - a one-shot scraper for a single project that is used for predicting over a running project and converting the output of the previous scraper to a better format
utils/stop_words.py - a list of words to ignore while extracting features
utils/deterministic_bag_of_words - a list of words we explicitly look for
utils/csv_creator.py - extract a category of projects from the external csv dataset
