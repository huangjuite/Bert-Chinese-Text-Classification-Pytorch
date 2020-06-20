import pandas as pd
import numpy as np
import re
import tqdm
import sys

train = pd.read_csv('train_data.csv')
print()
for ind in train.index:
    sys.stdout.write("\033[K")
    print('train',ind)
    sys.stdout.write("\033[F")
    train['title'][ind] = train['title'][ind].replace('\t', ' ')
train.to_csv('train_data.csv')

test = pd.read_csv('test_data.csv')
print()
for ind in test.index:
    sys.stdout.write("\033[K")
    print('test',ind)
    sys.stdout.write("\033[F")
    test['title'][ind] = test['title'][ind].replace('\t', ' ')
test.to_csv('test_data.csv')
