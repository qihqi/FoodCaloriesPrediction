from bs4 import BeautifulSoup
import requests
import os

START_DIR = '/home/han/Downloads/dataset/PFID_Lab_Stills/data'

all_num = set()
def get_numbers(l):
    result = []
    for i in l:
        try:
            x = int(i)
            result.append(x)
        except:
            pass
    return result


for current_dir, dirs, files in os.walk(START_DIR):
    for f in files:
        full_path = os.path.join(current_dir, f)
        elem = full_path.split('/')
        if len(elem) >= 2:
            num = get_numbers(elem)
            if len(num) == 2:
                all_num.add(tuple(num))

print(all_num)

URL = 'http://pfid.rit.albany.edu/viewer.php?rest_id={}&food_id={}&type=2'
results = {}
for (rest_id, fid) in all_num:
    page = requests.get(URL.format(rest_id, fid))
    soup = BeautifulSoup(page.text)
    div = soup.find("div", {"id": "main-body-left"})
    print(div.contents[3], div.contents[7], div.contents[11])
    results[(rest_id, fid)] = {
        'restaurant': div.contents[3],
        'food_name': div.contents[7],
        'calorie': div.contents[11],
        'raw': list(map(str, div.contents[:20]))
    }

print(results)
