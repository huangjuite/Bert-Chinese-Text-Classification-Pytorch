import pandas as pd
import numpy as np
from tqdm import tqdm
import os

try:
    os.mkdir('../dl_title_only/data')
    os.mkdir('../dl_with_key/data')
except:
    pass

class_name = ['news_entertainment',
              'news_sports',
              'news_house',
              'news_car',
              'news_edu',
              'news_tech',
              'news_military',
              'news_world',
              'news_agriculture',
              'news_game']
with open('../dl_title_only/data/class.txt', 'w') as f:
    for item in class_name:
        f.write("%s\n" % item)
with open('../dl_with_key/data/class.txt', 'w') as f:
    for item in class_name:
        f.write("%s\n" % item)

columns_titles = ["title","label"]

# # no keyword ------------------------------------------------------------
# train_txt = pd.read_csv('train_data.csv')
# train_txt = train_txt.drop(['ID', 'keyword', 'label_name'], axis=1)
# train_txt=train_txt.reindex(columns=columns_titles)
# train_txt.to_csv('../dl_title_only/data/train.txt', header=None, index=None, sep='\t', mode='w')

test_txt = pd.read_csv('test_data.csv')
test_txt = test_txt.drop(['id', 'keyword'], axis=1)
test_txt['label'] = 0
test_txt.to_csv('../dl_title_only/data/test.txt', header=None, index=None, sep='\t', mode='w')
test_txt.to_csv('../dl_title_only/data/dev.txt', header=None, index=None, sep='\t', mode='w')


# # with keyword ------------------------------------------------------------
# train_txt_key = pd.read_csv('train_data.csv')
# train_txt_key['title'] = train_txt_key[train_txt_key.columns[3:]].apply(
#     lambda x: ','.join(x.dropna().astype(str)),
#     axis=1
# )
# train_txt_key = train_txt_key.drop(['ID', 'label_name', 'keyword'], axis=1)
# train_txt_key=train_txt_key.reindex(columns=columns_titles)
# train_txt_key.to_csv('../dl_with_key/data/train.txt', header=None, index=None, sep='\t', mode='w')

test_txt_key = pd.read_csv('test_data.csv')
test_txt_key['title'] = test_txt_key[test_txt_key.columns[1:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)
test_txt_key = test_txt_key.drop(['id', 'keyword'], axis=1)
test_txt_key['label'] = 0
test_txt_key.to_csv('../dl_with_key/data/test.txt', header=None, index=None, sep='\t', mode='w')
test_txt_key.to_csv('../dl_with_key/data/dev.txt', header=None, index=None, sep='\t', mode='w')

