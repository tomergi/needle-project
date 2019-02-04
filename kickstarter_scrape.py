import pandas
import json
from datetime import datetime
import re
import time
import lxml
import lxml.html
from lxml.etree import tostring
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pickle
import csv

MAX_BACKERS = re.compile(r"Limited \([0-9]+ left of ([0-9]+)\)")

browser = webdriver.Firefox()


def parse_main_page():
    page_pattern = "https://www.kickstarter.com/discover/advanced?state=live&category_id=16&woe_id=0&raised=0&sort=end_date&seed=2576250&page=%d"
    current_page = 1
    browser.get(page_pattern % current_page)
    wait = WebDriverWait(browser, 5)
    wait.until(EC.presence_of_element_located((By.ID, "advanced_container")))
    items = []
    projects = set()
    try:
        while current_page < 200:
            r = lxml.html.fromstring(browser.page_source)
            prev_len = len(projects)
            for element in r.xpath("//a[contains(@class, 'soft-black mb3')]"):
                url = element.attrib['href']
                projects.add(url)
            try:
                browser.find_element_by_class_name('load_more').click()
                time.sleep(1)
            except:
                break
            current_page += 1
            pickle.dump(projects, open("projects_less_successful.pkl", "wb"))
    except:
        pass

    outputfd = open("result_less_successful.json", "w")
    outputfd.write("[\n")
    last_item = None
    for url in projects:
        item = parse_page(url)
        if item:
            if last_item:
                outputfd.write(json.dumps(last_item))
                outputfd.write(",\n")
                outputfd.flush()
            last_item = item
    if last_item:
        outputfd.write(json.dumps(last_item))
        outputfd.write("]")
        outputfd.flush()

        
def parse_page(url):
    try:
        print("parsing item", url)
        item = {}

        item['url'] = url
        browser.get(url)
        time.sleep(2)
        r = lxml.html.fromstring(browser.page_source)
        if "We apologize but something's gone wrong — an old link, a bad link, or some little glitch." in browser.page_source:
            return None
        try:
            title = r.xpath("//h2[contains(@class, 'type-24 type-28-sm type-38-md navy-700 medium mb3')]/text()")
            if len(title) == 0:
                title = r.xpath("//meta[contains(@property, 'og:title')]")[0].attrib['content']
            else:
                title = title[0]
            item["Title"] = title
        except:
            pass
        try:
            creator = r.xpath("//a[contains(@class, 'm0 p0 medium soft-black type-14 pointer keyboard-focusable')]/text()")
            if not creator:
                creator = re.match(r".* by (.*)\s+.[\s.]+[Kk]ickstarter",r.css('title::text')[0], re.DOTALL).group(1)
            else:
                creator = creator[0]
            item["Creator"] = creator
        except:
            pass
        allOrNothingText = "All or nothing."
        try:
            item["AllOrNothing"] = r.xpath("//a[contains(@class, 'soft-black pointer text-underline')]/text()")[0] == allOrNothingText
        except:
            pass
        try:
            item["DaysToGo"] = r.xpath("//span[contains(@class, 'block type-16 type-24-md medium soft-black')]/text()")[0]
        except:
            pass
        try:
            item["DollarsGoal"] = r.xpath("//span[contains(@class, 'money')]/text()")[0]
        except:
            pass
        try:
            item["DollarsPledged"] = r.xpath("//span[contains(@class, 'ksr-green-700')]/text()")[0]
        except:
            pass
        try:
            item["NumBackers"] = r.xpath("//div[contains(@class, 'block type-16 type-24-md medium soft-black')]//span/text()")[0]
        except:
            pass
        try:
            rewards = r.xpath("//div[contains(@class, 'NS_projects__rewards_list js-project-rewards')]/ol/li")
            #rewards = response.xpath("/html/body/main/div/div/div[2]/section[1]/div/div/div/div/div[2]/div[1]/div/ol/li")
            parsed_rewards = []
            i = 0
            for reward in rewards:
                try:
                    parsed_reward = {}
                    parsed_reward['id'] = i
                    root = lxml.html.fromstring(tostring(reward))
                    money = root.xpath("//span[contains(@class, 'money')]")
                    if money:
                        money = money[0].text
                    else:
                        money = ""
                    parsed_reward['Price'] = money
                    numbackers = root.xpath("//span[contains(@class, 'pledge__backer-count')]")
                    if numbackers:
                        numbackers = numbackers[0].text.strip()
                    else:
                        numbackers = ""
                    parsed_reward['NumBackers'] = numbackers
                    total_possible_backers = root.xpath("//span[contains(@class, 'pledge__limit')]")
                    if total_possible_backers:
                        total_possible_backers = total_possible_backers[0].text.strip()
                    else:
                        total_possible_backers = ""
                    m = MAX_BACKERS.match(total_possible_backers)
                    if m:
                        total_possible_backers = m.group(1)
                    if total_possible_backers == "Reward no longer available":
                        total_possible_backers = numbackers
                    parsed_reward['TotalPossibleBackers'] = total_possible_backers
                    parsed_reward['Text'] = tostring(reward).decode('utf-8')

                    if parsed_reward['Price']:
                        parsed_rewards.append(parsed_reward)
                        i += 1
                except:
                    pass
            item['rewards'] = parsed_rewards
        except:
            pass
        
        item["Text"] = tostring(r).decode('utf-8')

        return item
    except Exception as e:
        return None
def scrape_projects_list(projects):
    print("there are %d projects" % len(projects))
    outputfd = open("result.json", "w")
    outputfd.write("[\n")
    last_item = None
    for url, csv_row in projects:
        item = parse_page(url)
        if item:
            item['csv_row'] = csv_row
            if last_item:
                outputfd.write(json.dumps(last_item))
                outputfd.write(",\n")
                outputfd.flush()
            last_item = item
    if last_item:
        outputfd.write(json.dumps(last_item))
        outputfd.write("]")
        outputfd.flush()

def get_projects_list_from_csv(filename="ks-projects-201801.csv"):
    urls = []
    with open(filename, "r") as fd:
        reader = csv.reader(fd)
        for row in reader:
            try:
                project_id = int(row[0])
                project_name = row[1]
                project_url = "https://www.kickstarter.com/projects/%d/%s" % (project_id, project_name.replace("-", "").replace("!", "").replace("&","").replace(":","").replace("(","").replace(")","").replace(".","").replace(",","").replace("  "," ").replace("  ", " ").replace(" ", "-"))
                urls.append((project_url, row))
            except Exception as e:
                print(e)
                continue
    return urls

#projects = pickle.load(open("projects.pkl", "rb"))
#parse_main_page()
scrape_projects_list(get_projects_list_from_csv('food_dollars.csv'))
