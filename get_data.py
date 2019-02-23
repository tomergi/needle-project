import json
import numpy as np

def load_dataset(file_path, max_items):
    global successful_projects
    items = []
    pred = []
    with open(file_path, 'r') as fd:
        for row in fd:
            try:
                item = json.loads(row.strip()[:-1])
                _, csv_id, csv_name, csv_category, csv_main_category, csv_currency, csv_deadline, csv_goal, csv_launched, csv_pledged, csv_state, csv_backers, csv_country, csv_usd_pledged, csv_pledged_real, csv_usd_goal_real = item['csv_row']
                item['csv_id'] = csv_id
                item['csv_name'] = csv_name
                item['csv_category'] = csv_category
                item['csv_main_category'] = csv_main_category
                item['csv_currency'] = csv_currency
                item['csv_deadline'] = csv_deadline
                item['csv_goal'] = csv_goal
                item['csv_launched'] = csv_launched
                item['csv_pledged'] = csv_pledged
                item['csv_state'] = csv_state
                item['csv_backers'] = csv_backers
                item['csv_country'] = csv_country
                item['csv_usd_pledged'] = csv_usd_pledged
                item['csv_pledged_real'] = csv_pledged_real
                item['csv_usd_goal_real'] = float(csv_usd_goal_real)
                if item['csv_state'] == 'canceled':
                    continue
                items.append(item)
                if item['csv_state'] == 'successful':
                    pred.append(1)
                    successful_projects += 1
                else:
                    pred.append(0)
                if len(items) > max_items:
                    break
            except:
                continue

    return items, np.array(pred)