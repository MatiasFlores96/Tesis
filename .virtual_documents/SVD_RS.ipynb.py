import random

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader

from surprise.model_selection import GridSearchCV



import os
import pandas as pd
import numpy as np
from surprise.model_selection import train_test_split


# Cargamos el dataset nuestro
file_path = os.path.expanduser('ml-100k/u.data')

reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))

data = Dataset.load_from_file(file_path, reader=reader)



#Asignamos los datos a una lista raw_ratings, sin indices
raw_ratings = data.raw_ratings



random.shuffle(raw_ratings)


# Separamos en train y test
threshold = int(.9 * len(raw_ratings))
train_raw_ratings = raw_ratings[:threshold]
test_raw_ratings = raw_ratings[threshold:]

data.raw_ratings = train_raw_ratings 
# Reemplaza en data con los valores de entrenamiento


# Utilizamos gridsearch para obetener los mejores parametros para el algoritmo
print('Grid Search...')
param_grid = {'n_factors': [50,100,150],'n_epochs':[25,50,75],'lr_all': [0.005,0.01],'reg_all':[0.1,0.5,1]}
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs = -1)
grid_search.fit(data)


algo = grid_search.best_estimator['rmse']

print(grid_search.best_score['rmse'])
print(grid_search.best_params['rmse'])


print(algo)



# Se reentrena el set de entrenamiento con el mejor conjunto de parametros obtenido

trainset = data.build_full_trainset()
algo.fit(trainset)



# Con la primer funcion creamos un set de test a partir del set de entrenamiento
predictions = algo.test(trainset.build_testset())
print('Accuracy on Trainset,', end='   ')
accuracy.rmse(predictions)


# Prueba el modelo con el set de test
testset = data.construct_testset(test_raw_ratings)  
predictions = algo.test(testset)
print('Accuracy on Testset,', end=' ')
accuracy.rmse(predictions)


print(trainset.n_users)
print(trainset.n_items)
print(algo.qi.shape)
print(algo.pu.shape)


#predictions (Tiene las predicciones para un usuario y un item, el rating reak y el estimado)


from collections import defaultdict
import os
import io


def get_top_n(predictions, n = 10):
    
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
        
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n


top_n = 10
top_pred = get_top_n(predictions, n = top_n)



def read_item_names(file_path):
    rid_to_name = {}
    name_to_rid = {}
    
    with io.open(file_path, 'r', encoding = 'ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]
    
    return rid_to_name, name_to_rid


item_filepath = 'ml-100k/u.item'
rid_to_name, name_to_rid = read_item_names(item_filepath)





# User raw Id
uid_list = ['4']

# Imprime las 10 recomendaciones de peliculas para un determinado usuario
for uid, user_ratings in top_pred.items():
    if uid in uid_list:
        for (iid, rating) in user_ratings:
            movie = rid_to_name[iid]
            print('Movie:', iid, '-', movie, ', rating:', str(rating))



uid_list = ['697']
for uid in predictions:
    if uid in uid_list:
        for (iid, rating) in user_ratings:
            movie = rid_to_name[iid]
            print('Movie:', iid, '-', movie, ', rating:', str(rating))






