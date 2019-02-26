from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
from time import sleep
import re
import random
import base64


CURRENCY_CONVERSION_DICT = {
    '$': 1.0,
    '€': 1.13,
    '£': 1.31,
}


CURRENT_DIR = os.path.split(__file__)[0]

OUTPUT_JSON_FILE = os.path.join(CURRENT_DIR, 'output.json')
MIN_WAIT_TIME, MAX_WAIT_TIME = (5, 5)

# This supposed to capture (Updates <Num>) | (Comments <Num>)
RE_TABULAR_SECTION = re.compile("\s+(?:[A-Za-z]+)\s+(\d+)\s+")
PARSE_MONEY_STR = "([$¢£¤¥֏؋৲৳৻૱௹฿៛\u20a0-\u20bd\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6])([\d,]+)"
RE_PARSE_MONEY = re.compile(PARSE_MONEY_STR)
# RE_GOAL = re.compile("pledged\sof\s"+PARSE_MONEY_STR+"\sgoal")
RE_PLEDGED = re.compile("\$\s*([\d,]+)")
RE_BACKERS = re.compile("([\d,]+)\s*backers")
RE_TIME_LEFT = re.compile("(\w+)\s*(\w+)(?:to\sgo)")


COUNTRY_DATAFRAME = pd.read_csv("./world-cities.csv")
# def parse_card(soup):
#     attrs = {}
#     stats_pat = re.compile('(.*)\s+made\s+it\s+\|\s+(.*)\s+reviews\s+\|\s+(.*)\s+photos')
#     stats = soup.find_all('div', {'class': 'summary-stats-box'})[0].text.replace('\n', '')
#     m = re.match(stats_pat, stats)
#     if m:
#         attrs['NumMadeIt'] = m.group(1)
#         attrs['NumReviews'] = m.group(2)
#         attrs['NumPhotos'] = m.group(3)
#     else:
#         raise ValueError('Could not find stats bar')
#     attrs['Ingredients'] = [s.text for s in soup.find_all('span', {'class': "recipe-ingred_txt added"})]
#     attrs['Directions'] = [s.text.strip() for s in soup.find_all('span', {'class': "recipe-directions__list--item"}) if len(s) != 0]
#
#     times = soup.find('ul', {'class': 'prepTime'}).find_all('time')
#     attrs['PrepTime'] = times[0].text
#     attrs['CookTime'] = times[1].text
#     attrs['ReadyIn'] = times[2].text
#     return attrs
#
#
# def parse_thumbnail(soup):
#     attrs = {}
#     # Title, Creator, Rating,
#     attrs['Title'] = soup.find('span', {'class': 'fixed-recipe-card__title-link'}).text
#     attrs['Creator'] = soup.text[soup.text.find("By")+3:].rstrip()
#
#     ratings_element = soup.find_all('div', {'class': 'fixed-recipe-card__ratings'})[0]
#     attrs['Rating'] = ratings_element.find('span')['data-ratingstars']
#
#     recipe_link = soup.find_all('a', {'class': 'fixed-recipe-card__title-link'})[0]['href']
#     recipe_soup = load_page(recipe_link)
#
#     more_attrs = parse_card(recipe_soup)
#     attrs.update(more_attrs)
#     return attrs


def load_page(url):
    rand_time = random.randint(MIN_WAIT_TIME, MAX_WAIT_TIME)
    sleep(rand_time)
    with urllib.request.urlopen(url) as response:
        raw_html = response.read()
    soup = BeautifulSoup(raw_html, 'html.parser')
    return soup


def extract_title_and_description(soup):
    title_desc_cont = soup.find_all('div', attrs={"class": "col-20-24 block-md order-2-md col-lg-14-24"})
    elems = list(title_desc_cont[0].find_all())[:-1]
    title = elems[0].text
    description = elems[1].text
    return title, description


def extract_num_updates(soup):
    num_updates_text = soup.find_all('a', attrs={'class': 'js-load-project-updates'})[0].text
    m = RE_TABULAR_SECTION.match(num_updates_text)
    if m:
        num_updates = m.group(1)
        return num_updates
    else:
        raise ValueError("Could not find number of updates!")


def extract_num_comments(soup):
    num_comments_text = soup.find_all('a', attrs={'class': 'js-load-project-comments'})[0].text
    m = RE_TABULAR_SECTION.match(num_comments_text)
    if m:
        num_comments = m.group(1)
        return num_comments
    else:
        raise ValueError("Could not find number of comments!")


def extract_country_subcountry_city(soup):
    filter_lambda = lambda s: s.startswith("https://www.kickstarter.com/discover/places/")
    cand_city, cand_subcountry = tuple(map(str.strip, soup.find_all('a', attrs={'href': filter_lambda})[0].text.split(',')))
    result = COUNTRY_DATAFRAME.loc[COUNTRY_DATAFRAME['name'] == cand_city]

    city = result['name'].values[0]
    subcountry = result['subcountry'].values[0]
    country = result['country'].values[0]
    return country, subcountry, city

def extract_category_subcategory(soup):
    filter_lambda = lambda s: s.startswith("https://www.kickstarter.com/discover/categories/")
    result = soup.find_all('a', attrs={'href': filter_lambda})[0]
    return result.text

def extract_creator(soup):
    creator = soup.find_all('span', attrs={'class': "soft-black ml2 ml0-md break-word"})[0].text
    return creator


def extract_num_projects_by_creator(soup):
    num_projects_text = soup.find_all('a', attrs={'class': "dark-grey-500 keyboard-focusable"})[0].text  # "<Num> created"
    end_idx = num_projects_text.index(' ')
    return num_projects_text[0:end_idx]


def parse_money(raw_text):
    # Sometimes it loads differently like in those 2 cases:
    if raw_text[0] in CURRENCY_CONVERSION_DICT:  # 'pledged of €20,000 goal'
        currency, amount = raw_text[0], raw_text[1:]
    elif raw_text[-1] in CURRENCY_CONVERSION_DICT:
        raw_text = raw_text.replace('.', ',')  # 'pledged of 20.000€goal'
        currency, amount = raw_text[-1], raw_text[0:-1]
    else:
        raise ValueError("Error with parsing of money: %s" % raw_text)
    amount = amount.replace(',', '')
    amount = float(amount)
    if currency in CURRENCY_CONVERSION_DICT:
        amount = CURRENCY_CONVERSION_DICT[currency] * amount
    else:
        raise ValueError("Non supported currency based project")
    return np.round(amount, decimals=2)

def extract_goal(soup):
    goal_text = soup.find_all('span', attrs={'class': "inline-block-sm hide"})[0].text
    # 'pledged of €20,000 goal'
    if goal_text.startswith("pledged of ") and goal_text.endswith("goal"):
        goal_text = goal_text.replace("pledged of ", "").replace("goal", "").replace(".", ",").strip()
        return parse_money(goal_text)
    else:
        raise ValueError("Could not find goal! ('pledged of...).")
    # m = RE_GOAL.match(goal_text)
    # if m:
    #     currency = m.group(1)
    #     amount = m.group(2)
    #     return parse_money(currency, amount)
    # else:
    #     raise ValueError("Could not find goal! ('pledged of...).")

def extract_num_pledged(soup):
    num_pledged_text = soup.find_all("span", attrs={'class': "ksr-green-700 inline-block medium type-16 type-24-md"})[0].text
    m = RE_PARSE_MONEY.match(num_pledged_text)
    if m:
        return parse_money(num_pledged_text)
    else:
        raise ValueError("Could not find pledged amount.")


def extract_num_backers(soup):
    num_backers_text = soup.find_all('div', attrs={'class': "ml5 ml0-lg mb2-lg"})[0].text
    m = RE_BACKERS.match(num_backers_text)
    if m:
        return m.group(1)
    else:
        raise ValueError("Could not find number of backers!")


def extract_time_left(soup):
    elem = soup.find_all('p', attrs={"class": "mb3 mb0-lg type-12"})[0]
    return elem.text


def extract_project_we_love(soup):
    result = soup.find_all('div', attrs={'class': "py2 py3-lg flex items-center auto-scroll-x"})[0]
    children = result.find_all('span', attrs={"class": "ml1"})
    for c in children:
        if "Project We Love" in c.text():
            return True
    return False


def parse_page(soup):
    item = {}
    item['title'], item['description'] = extract_title_and_description(soup)
    item['num_updates'] = extract_num_updates(soup)
    item['num_comments'] = extract_num_comments(soup)
    item['country'], item['subcountry'], item['city'] = extract_country_subcountry_city(soup)
    item['creator'] = extract_creator(soup)
    item['num_projects_by_creator'] = extract_num_projects_by_creator(soup)
    item['num_backers'] = extract_num_backers(soup)
    item['num_pledged'] = extract_num_pledged(soup)
    item['goal'] = extract_goal(soup)
    item['timeleft'] = extract_time_left(soup)
    item['project_we_love'] = extract_project_we_love(soup)
    item['category'], item['subcategory'] = extract_category_subcategory(soup)
    item['is_successful'] = extract_is_successful(soup)
    item['rewards'] = extract_rewards(soup)
    return item


def scrap_from_file(fp):
    pass


REWARD_PRICE_RE = re.compile(r".*[\d,.]+.*")
def extract_from_json_line(json_line):
    raw_html = json_line['Text'].decode('utf-8')
    soup = BeautifulSoup(raw_html, 'html.parser')
    item = {}
    item['url'] = json_line['url']
    item['title'] = json_line['Title']
    item['creator'] = json_line['Creator']
    item['timeleft'] = json_line['DaysToGo']
    csv_row = json_line['csv_row']
    index, ID,name,category,main_category,currency,deadline,goal,launched,pledged,state,backers,country,usd_pledged,usd_pledged_real,usd_goal_real = csv_row
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


    item['title'], item['description'] = extract_title_and_description(soup)
    item['num_updates'] = extract_num_updates(soup)
    item['num_comments'] = extract_num_comments(soup)
    item['country'], item['subcountry'], item['city'] = extract_country_subcountry_city(soup)
    item['creator'] = extract_creator(soup)
    item['num_projects_by_creator'] = extract_num_projects_by_creator(soup)
    item['num_backers'] = extract_num_backers(soup)
    item['num_pledged'] = extract_num_pledged(soup)
    item['goal'] = extract_goal(soup)
    item['timeleft'] = extract_time_left(soup)
    item['project_we_love'] = extract_project_we_love(soup)
    item['category'], item['subcategory'] = extract_category_subcategory(soup)
    #item['is_successful'] = extract_is_successful(soup)
    #item['rewards'] = extract_rewards(soup)
    return item
    

# def run(base_url):
#     all_items = {}
#     try:
#         max_page = 100
#         for page in range(1, max_page):  # Start with page 1
#             print('Page: %d / %d' % (page, max_page))
#             cur_url = base_url + '?page=%d' % page
#             soup = load_page(cur_url)
#             parsed = parse_page(soup)
#             all_items.update(parsed)
#             if len(all_items) >= MIN_NUM_ITEMS:
#                 break
#             else:
#                 print("Current number of recipes: %d" % len(all_items))
#     except Exception as e:
#         pass
#     if len(all_items) > 0:
#         with open(OUTPUT_JSON_FILE, 'w') as fp:
#             json.dump(all_items, fp, indent=4)


if __name__ == '__main__':
    # run(base_url=base_url)
    link1 = "https://www.kickstarter.com/projects/13/1000056157"
    link2 = "https://www.kickstarter.com/projects/horriblegames/alonetm-2nd-print-run-with-new-contents-and-locali?ref=discovery_category"
    soup = load_page(link2)
    parsed = parse_page(soup)
