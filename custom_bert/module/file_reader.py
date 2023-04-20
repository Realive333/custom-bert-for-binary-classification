import json
import pandas as pd
from tqdm import tqdm

def getDataframeByPath(path, length):
    with open(path) as f:
        works = json.load(f)
    datas = []
    for work in tqdm(works):
        label = work['label']
        text = [c['paragraph'] for c in work['contents']]
        try:
            assert len(text) == length, f'Work contents length should be {length}, is {len(text)}'
            datas.append({'label': label, 'text': text})
        except AssertionError as err:
            print(err)
    return pd.DataFrame.from_dict(datas)