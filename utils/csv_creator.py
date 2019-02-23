import pandas
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("main", help="main category to scrape")
    args = parser.parse_args()
    csv = pandas.read_csv("ks-projects-201801.csv")
    main_category = csv[csv['main_category'] == args.main]
    usd_main_category = main_category[main_category['currency'] == 'USD']
    name = "usd_" + args.main + ".csv"
    usd_main_category.to_csv(name)
