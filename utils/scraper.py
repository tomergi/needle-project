from bs4 import BeautifulSoup
import os
import json
import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
from time import sleep
import re
import random
import base64
import argparse
from datetime import datetime
import lxml
import lxml.html


CURRENCY_CONVERSION_DICT = {
    '$': 1.0,
    '€': 1.13,
    '£': 1.31,
}


CURRENT_DIR = os.path.split(__file__)[0]

MIN_WAIT_TIME, MAX_WAIT_TIME = (5, 5)

# This supposed to capture (Updates <Num>) | (Comments <Num>)
RE_TABULAR_SECTION = re.compile("\s+(?:[A-Za-z]+)\s+([\d,]+)\s+")
PARSE_MONEY_STR = "([$¢£¤¥֏؋৲৳৻૱௹฿៛\u20a0-\u20bd\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6])([\d,]+)"
RE_PARSE_MONEY = re.compile(PARSE_MONEY_STR)
# RE_GOAL = re.compile("pledged\sof\s"+PARSE_MONEY_STR+"\sgoal")
RE_PLEDGED = re.compile("\$\s*([\d,]+)")
RE_BACKERS = re.compile("([\d,]+)\s*backers?")
RE_TIME_LEFT = re.compile("(\w+)\s*(\w+)(?:to\sgo)")

def load_page(url):
    rand_time = random.randint(MIN_WAIT_TIME, MAX_WAIT_TIME)
    sleep(rand_time)
    with urllib.request.urlopen(url) as response:
        raw_html = response.read()
    soup = BeautifulSoup(raw_html, 'html.parser')
    return soup


def extract_rewards(soup):
    rewards = soup.find_all("h2", attrs={'class': 'pledge__amount'})
    parsed_rewards = []
    i = 0
    for reward in rewards:
        try:
            parsed_reward = {}
            parsed_reward['id'] = i
            root = lxml.html.fromstring(str(reward))
            converted_money = reward.find_all("span", attrs={'class' : 'pledge__currency-conversion'})
            if converted_money:
                money = converted_money[0].text
            else:
                money = root.xpath("//span[contains(@class, 'money')]")
                if money:
                    money = money[0].text
                else:
                    money = ""
            price = REWARD_PRICE_RE.match(money.strip().replace(",", ""))
            parsed_reward['price'] = float(price.group(1))
            if parsed_reward['price']:
                parsed_rewards.append(parsed_reward)
                i += 1
        except:
            continue
    return parsed_rewards

#NOTE: successful projects have different layout than running/failed projects

def extract_title_and_description(soup):
    """Get the title and description of a project"""
    title_desc_cont = soup.find_all('div', attrs={"class": "col-20-24 block-md order-2-md col-lg-14-24"})
    if len(title_desc_cont) > 0:  # if unsuccessful
        elems = list(title_desc_cont[0].find_all())[:-1]
        title = elems[0].text
        description = elems[1].text
    else:  # if successful
        title = soup.find_all('div', attrs={"class": "NS_project_profile__title"})[0].text.strip()
        description = soup.find_all('div', attrs={"class": "NS_project_profiles__blurb"})[0].text.strip()
    return title, description


def extract_num_updates(soup):
    """Get the number of updates the creator published for this project"""
    num_updates_text = soup.find_all('a', attrs={'class': 'js-load-project-updates'})[0].text
    m = RE_TABULAR_SECTION.match(num_updates_text)
    if m:
        num_updates = m.group(1).replace(',', '')
        return int(num_updates)
    else:
        raise ValueError("Could not find number of updates!")


def extract_num_comments(soup):
    """Get the number of comments on this project"""
    num_comments_text = soup.find_all('a', attrs={'class': 'js-load-project-comments'})[0].text
    m = RE_TABULAR_SECTION.match(num_comments_text)
    if m:
        num_comments = m.group(1).replace(',', '')
        return int(num_comments)
    else:
        raise ValueError("Could not find number of comments!")


def extract_category_subcategory(soup):
    """Get the category and subcategory of a project"""
    category_re = r".*discover/categories/([0-9a-zA-Z ]+)/"
    filter_lambda = lambda s: s.startswith("https://www.kickstarter.com/discover/categories/")
    result = soup.find_all('a', attrs={'href': filter_lambda})[0]
    match = re.match(category_re, result['href'])
    category = match.group(1).capitalize()

    return category, result.text

def extract_creator(soup):
    """Get the name of the creator"""
    creator = soup.find_all('span', attrs={'class': "soft-black ml2 ml0-md break-word"})
    if len(creator) == 0:  # if project unsuccesful
        creator = soup.find_all('div', attrs={"class": "creator-name"})[0].text.strip()
        creator = creator[:creator.index('\n')]
    else:  # if successful
        creator = creator[0].text
    return creator



def parse_money(raw_text):
    """convert a money string into dollars"""
    # Sometimes it loads differently like in those 2 cases:
    if raw_text[0] in CURRENCY_CONVERSION_DICT:  # 'pledged of €20,000 goal'
        currency, amount = raw_text[0], raw_text[1:]
    elif raw_text[-1] in CURRENCY_CONVERSION_DICT:
        raw_text = raw_text.replace('.', ',')  # 'pledged of 20.000€goal'
        currency, amount = raw_text[-1], raw_text[0:-1]
    else:
        raise ValueError("Error with parsing of money: %s" % raw_text)
    amount = amount.replace(',', '').replace(' ', '')
    amount = float(amount)
    if currency in CURRENCY_CONVERSION_DICT:
        amount = CURRENCY_CONVERSION_DICT[currency] * amount
    else:
        raise ValueError("Non supported currency based project")
    return np.round(amount, decimals=2)

def extract_goal(soup):
    """Get the USD goal of a project"""
    goal_text = soup.find_all('span', attrs={'class': "inline-block-sm hide"})[0].text
    # 'pledged of €20,000 goal'
    if goal_text.startswith("pledged of ") and goal_text.endswith("goal"):
        goal_text = goal_text.replace("pledged of ", "").replace("goal", "").replace(".", ",").strip()
        return parse_money(goal_text)
    else:
        raise ValueError("Could not find goal! ('pledged of...).")

def extract_num_pledged(soup):
    """Get the amount of money people pledged for the project"""
    num_pledged_text = soup.find_all("span", attrs={'class': "ksr-green-700 inline-block medium type-16 type-24-md"})[0].text
    m = RE_PARSE_MONEY.match(num_pledged_text)
    if m:
        return parse_money(num_pledged_text)
    else:
        raise ValueError("Could not find pledged amount.")


def extract_num_backers(soup):
    """Get the number of backers for a project"""
    num_backers_text = soup.find_all('div', attrs={'class': "ml5 ml0-lg mb2-lg"})[0].text
    m = RE_BACKERS.match(num_backers_text)
    if m:
        return m.group(1)
    else:
        raise ValueError("Could not find number of backers!")


def extract_time_left(soup):
    """Get the time left until the project ends"""
    elem = soup.find_all('p', attrs={"class": "mb3 mb0-lg type-12"})[0]
    return elem.text


def extract_about(soup):
    """Get the about section of a project"""
    result = soup.find_all('div', attrs={"class": "full-description js-full-description responsive-media formatted-lists"})
    return result[0].text.strip()

def extract_hoursduration(soup):
    """Get the amount of hours this project lasts"""
    elem = soup.find_all('span', attrs={"class": "block type-16 type-24-md medium soft-black"})[0]
    return float(elem.text) * 24 * 3600



def parse_page(soup):
    """parse the html of a page and return the parsed item"""
    item = {}
    item['title'], item['description'] = extract_title_and_description(soup)
    item['num_updates'] = extract_num_updates(soup)
    item['num_comments'] = extract_num_comments(soup)
    item['creator'] = extract_creator(soup)
    item['num_backers'] = extract_num_backers(soup)
    item['num_pledged'] = extract_num_pledged(soup)
    item['goal'] = extract_goal(soup)
    item['category'], item['subcategory'] = extract_category_subcategory(soup)
    item['about'] = extract_about(soup)
    item['timeleft'] = extract_time_left(soup)
    item['rewards'] = extract_rewards(soup)
    item['hours_duration'] = extract_hoursduration(soup)
    return item


def scrap_from_file(input_file_path, output_file_path):
    """Convert the input json that was generated by kickstarter_scrape into a classifier ready format and store it in output_file_path"""
    with open(input_file_path, mode='r') as inp_fp, open(output_file_path, mode='w') as out_fp:
        last_result = None
        for i, line in enumerate(inp_fp, 1):
            if line.strip() == '{':
                continue
            try:
                print("\rCurrently %d" % i, end="")
                dict_item = json.loads(line.strip()[:-1])
                result = extract_from_json_line(dict_item)

                if not last_result and result:
                    out_fp.write('[\n')

                if last_result:
                    out_fp.write(json.dumps(last_result))
                    out_fp.write(",\n")
                last_result = result
            except Exception as e:
                print("\n\turl:", dict_item['url'])
                continue

        if last_result:
            out_fp.write(json.dumps(last_result))
            out_fp.write("\n}")


REWARD_PRICE_RE = re.compile(r".*(?:US)?\$\s*([\d,.]+)\s*")
def extract_from_json_line(json_line):
    """Given a line from the json created by kickstarter_scrape, convert it to a classifier-ready format and add missing stuff"""
    raw_html = base64.b64decode(json_line['Text']).decode('utf-8')
    soup = BeautifulSoup(raw_html, 'html.parser')
    item = {}
    item['url'] = json_line['url']
    item['title'] = json_line['Title']
    if 'DaysToGo' in json_line:
        item['timeleft'] = json_line['DaysToGo']
    else:
        item['timeleft'] = 0
    csv_row = json_line['csv_row']
    index, ID,name,category,main_category,currency,deadline,goal,\
        launched,pledged,state,backers,country,usd_pledged,usd_pledged_real,usd_goal_real = csv_row
    # item['country'], item['subcountry'], item['city'] = extract_country_subcountry_city(soup)
    start = datetime.strptime(launched, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(deadline, "%Y-%m-%d")
    td = (end - start).total_seconds()
    td /= 3600  # in hours
    item['hours_duration'] = td
    item['state'] = state
    item['goal'] = usd_goal_real
    item['category'] = main_category
    item['subcategory'] = category
    rewards = []
    for reward in json_line['rewards']:
        reward_dict = {}
        reward_dict['id'] = reward['id']
        match = REWARD_PRICE_RE.match(reward['Price'])
        if not match:
            print("could not find reward price in string:", reward['Price'])
        else:
            reward_dict['price'] = float(match.group(1).replace(",",""))
        rewards.append(reward_dict)
    item['rewards'] = rewards
    item['num_pledged'] = float(pledged)

    item['title'], item['description'] = extract_title_and_description(soup)
    item['num_updates'] = extract_num_updates(soup)
    item['about'] = extract_about(soup)
    item['num_comments'] = extract_num_comments(soup)
    item['creator'] = extract_creator(soup)
    item['num_backers'] = backers
    return item
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("A kickstarter scraper that doesn't use selenium. Useful for individual pages. Outputs a JSON ready to use for classification")
    parser.add_argument("--scraped_json", action="store_true", help="input is a scraped json using kickstarter_scrape")
    parser.add_argument("input", help="input url or file")
    parser.add_argument("output", help="file to write with processed data")
    args = parser.parse_args()

    if args.scraped_json:
        scrap_from_file(args.input, args.output)
    else:
        soup = load_page(args.input)
        parsed = parse_page(soup)
        parsed['url'] = args.input
        json_out = json.dumps([parsed])
        with open(args.output, "w") as outfd:
            outfd.write(json_out)
