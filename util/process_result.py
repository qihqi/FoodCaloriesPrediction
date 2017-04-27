import sys
import os

START_DIR = '/home/han/Downloads/dataset/PFID_Lab_Stills/data'
with open('result.py') as f:
    result = eval(f.read())

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
                x = result[tuple(num)]
                print(repr(full_path),x['calorie'])


