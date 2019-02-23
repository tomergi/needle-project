from bs4 import BeautifulSoup
import os
import json
import urllib.request, urllib.parse
from time import sleep
import re
import random

CURRENT_DIR = os.path.split(__file__)[0]

OUTPUT_JSON_FILE = os.path.join(CURRENT_DIR, 'output.json')
MIN_WAIT_TIME, MAX_WAIT_TIME = (5, 5)

# This supposed to capture (Updates <Num>) | (Comments <Num>)
RE_TABULAR_SECTION = re.compile("\s+(?:[A-Za-z]+)\s+(\d+)\s+")
RE_GOAL = re.compile("pledged\sof\s\$([\d,]+)\sgoal")
RE_PLEDGED = re.compile("\$\s*([\d,]+)")
RE_BACKERS = re.compile("([\d,]+)\s*backers")
RE_TIME_LEFT = re.compile("(\w+)\s*(\w+)(?:to\sgo)")

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
        html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    return soup


def extract_title(soup):
    return soup.find_all('h2', attrs={'class': 'type-24 type-28-sm type-38-md navy-700 medium mb3'})[0].text


def extract_description(soup):
    return None

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


def extract_usa_state(soup):
    filter_lambda = lambda s: s.startswith("https://www.kickstarter.com/discover/places/")
    state_text = soup.find_all('a', attrs={'href': filter_lambda})[0].text
    return state_text


def extract_creator(soup):
    creator = soup.find_all('span', attrs={'class': "soft-black ml2 ml0-md break-word"})[0].text
    return creator


def extract_num_projects_by_creator(soup):
    num_projects_text = soup.find_all('a', attrs={'class': "dark-grey-500 keyboard-focusable"})[0].text  # "<Num> created"
    end_idx = num_projects_text.index(' ')
    return num_projects_text[0:end_idx]



def extract_goal(soup):
    goal_text = soup.find_all('span', attrs={'class': "inline-block-sm hide"})[0].text
    m = RE_GOAL.match(goal_text)
    if m:
        return m.group(1)
    else:
        raise ValueError("Could not find goal! ('pledged of...)")


def extract_num_backers(soup):
    num_backers_text = soup.find_all('div', attrs={'class': "ml5 ml0-lg mb2-lg"})[0].text
    m = RE_BACKERS.match(num_backers_text)
    if m:
        return m.group(1)
    else:
        raise ValueError("Could not find number of backers!")


def extract_num_pledged(soup):
    num_pledged_text = soup.find_all("span", attrs={'class': "ksr-green-700 inline-block medium type-16 type-24-md"})[0].text
    m = RE_PLEDGED.match(num_pledged_text)
    if m:
        return m.group(1)
    else:
        raise ValueError("Could not find pledged amount.")


def extract_time_left(soup):
    elem = soup.find_all('div', attrs={"class": "ml5 ml0-lg"})
    return elem.text


def parse_page(soup):
    item = {}
    item['title'] = extract_title(soup)
    item['desc'] = extract_description(soup)
    item['num_updates'] = extract_num_updates(soup)
    item['num_comments'] = extract_num_comments(soup)
    item['usa_state'] = extract_usa_state(soup)
    item['creator'] = extract_creator(soup)
    item['num_projects_by_creator'] = extract_num_projects_by_creator(soup)
    item['num_backers'] = extract_num_backers(soup)
    item['goal'] = extract_goal(soup)
    item['num_pledged'] = extract_num_pledged(soup)
    item['time_left'] = extract_time_left(soup)


    # num_recipes = len(recipe_cards)
    # for i, x in enumerate(recipe_cards, 1):
    #     print('\r\tRecipe: %d / %d' % (i, num_recipes), end='')
    #     try:
    #         parsed = parse_thumbnail(x)
    #         items[parsed['Title']] = parsed
    #     except Exception as e:
    #         pass
    #         # print("Failed to parse card. Thrown %s" % e)
    # print()  # To make a new line and reset the carriage return
    return item


def scrap_from_file(fp):
    pass


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
    soup = load_page("https://www.kickstarter.com/projects/13/1000056157")
    parsed = parse_page(soup)
