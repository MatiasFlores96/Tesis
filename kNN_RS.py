import os
import io
import numpy as np
from collections import defaultdict

from surprise import KNNBasic
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import cross_validate

# matplotlib inline
import matplotlib.pyplot as plt

file_path = os.path.expanduser('~/PycharmProjects/Tesis1/ml-100k/u.data')

reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))

data = Dataset.load_from_file(file_path, reader=reader)

kk = 50
sim_options = {'name': 'cosine', 'user_based': True}
algo = KNNBasic(k=kk, sim_options=sim_options, verbose=True)

cv = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


rmse = cv['test_rmse']
mae = cv['test_mae']
x = np.arange(len(rmse))

fig, ax = plt.subplots(figsize = (10, 5))
plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
plt.ylim(0.5, 1.3)
ax.plot(x, rmse, marker='o', label='rmse')
ax.plot(x, mae, marker='o', label='mae')

plt.title('Model Errors', fontsize=12)
plt.xlabel('cv', fontsize=10)
plt.ylabel('Error', fontsize=10)
plt.legend()
plt.show()


def read_item_names(file_path):
    rid_to_name = {}
    name_to_rid = {}

    with io.open(file_path, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid

item_filepath = '~/PycharmProjects/Tesis1/ml-100k/u.item'
rid_to_name, name_to_rid = read_item_names(item_filepath)