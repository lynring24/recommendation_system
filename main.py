#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def read_user_id():
    with open('./input.txt', 'r') as f:
        return [l.strip().split(',') for l in  f.readlines()]


def write_output(prediction):
    with open('./output.txt', 'w') as f:
        for pred in prediction:
            f.write(pred+"\n")    
    
    
def do(inputs):
    string_results = []
    for user, movie in inputs:
        key_user = int(user)
        key_movie = int(movie)
        print("(user, movie) = (%d, %d)"%(key_user, key_movie))
        string_results.append(task1(key_user, key_movie))
        string_results.append(task2(key_user, key_movie))
        string_results.append(task3(key_user, key_movie))       
    return string_results


def is_in_trainset(user, movie):
     return ((trainset['userId'] == user) & (trainset['movieId'] == movie)).any()
      

def task1(user, movie):
    prediction_cf = task1_predict(similarity, user)
    prediction = str(prediction_cf.round(4).loc[movie]['prediction'])
    if is_in_trainset(user, movie):
        print("Task1 RMSE: ", task1_RMSE(prediction_cf, user))
    else: 
        print("Task3 RMSE : ", task1_RMSE(prediction_cf, user, 'test'))
    return ','.join([str(user), str(movie), prediction])


def task2(user, movie):
    prediction_mf = decomposistion[int(user)]
    prediction = str(prediction_mf.round(4).loc[movie])
    if is_in_trainset(user, movie):
        print("Task2 RMSE : ", task2_RMSE(user))
    else :
        print("Task2 RMSE : ", task2_RMSE(user, 'test'))
    return ','.join([str(user), str(movie), prediction]) 


def task3(user, movie):
    prediction_mf2 = opt_decomposistion[int(user)]
    prediction = str(prediction_mf2.round(4).loc[movie])
    if is_in_trainset(user, movie):
        print("Task3 RMSE : ", task3_RMSE(user))
    else :
        print("Task3 RMSE : ", task3_RMSE(user, 'test'))
    return ','.join([str(user), str(movie), prediction])  
   
    
def initialize_train_data():
    global movieIds, userIds
    trainset = pd.read_csv('data/ratings_train.csv', usecols = ['userId', 'movieId', 'rating'])
    movieIds = sorted(trainset['movieId'].unique())
    userIds = sorted(trainset['userId'].unique())
    df_train = pd.DataFrame(index=sorted(movieIds), columns=sorted(userIds) )
    for index, rows in  trainset.iterrows():
        df_train.loc[rows['movieId']][rows['userId']] = rows['rating']
    return  trainset , df_train 


def initialize_test_data():
    targets = pd.read_csv('data/ratings_test.csv', usecols = ['userId', 'movieId', 'rating'])
    return targets


def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())


def task1_predict(similarity, userId):
#     movieIds = sorted(trainset['movieId'].unique())
    similarity = pd.DataFrame(similarity, index=movieIds, columns=movieIds)
    rated_movies = trainset[trainset.userId == userId]['movieId'].to_frame()
    user_sim = similarity.loc[rated_movies.movieId]
    user_rating = trainset[trainset.userId == userId]['rating'].to_frame()
    sim_sum = user_sim.sum(axis=0).to_frame()
    prediction = np.matmul(user_sim.T.to_numpy(), user_rating.to_numpy())/(sim_sum.to_numpy() + 1)
    prediction.round(4)
    prediction = pd.DataFrame(prediction, index=sorted(movieIds), columns=['prediction'])
    return prediction 


# for user uid, get prediction test
def task1_RMSE(prediction_cf, uid, mode='train'):
    rmse = 0 
    targets = []
    prediction = []
    if mode=='train':
        dataset = trainset[trainset.userId==uid] 
    else:
        dataset = df_target[df_target.userId==uid]
    for index, rows in dataset.iterrows():
        try: 
            prediction.append(prediction_cf.loc[rows['movieId']]['prediction'])     
            targets.append(rows['rating'])
        except KeyError:
            pass
        
    rmse  = RMSE(np.asarray(prediction), np.asarray(targets))    
    return rmse 


def build_svd(model, K=400):
    # task 2 
    # fill 0 by the mean values of movies
    filled_model= model.apply(lambda row: row.fillna(row.mean()), axis=1)
    # df_trainshape
    u, s, vh = np.linalg.svd(filled_model, True)
    u = u[:,:K]
    Sigma = np.diag(s[:K])
    vh = vh[:K, :]

    user_factors =  np.matmul(u, np.sqrt(Sigma))
    item_factors =  np.matmul(np.sqrt(Sigma),vh)
    df_prediction = pd.DataFrame(np.matmul(user_factors, item_factors),index=model.index, columns=model.columns)
    df_prediction = df_prediction.round(4)
    return df_prediction


# for user uid, get prediction test
def task2_RMSE(uid, mode='train'):
    rmse  = 0 
    targets = []
    prediction = []
      
    if mode=='train':
        dataset = trainset[trainset.userId==uid] 
    else:
        dataset = df_target[df_target.userId==uid]
    for index, rows in dataset.iterrows():
        try: 
            prediction.append(decomposistion.loc[int(rows['movieId']),int(rows['userId'])])    
            targets.append(rows['rating'])
        except KeyError:
            pass
    rmse = RMSE(np.asarray(prediction), np.asarray(targets))
    return rmse 


# for user uid, get prediction test
def task3_RMSE(uid, mode='train'):
    rmse  = 0 
    targets = []
    prediction = []
    if mode=='train':
        dataset = trainset[trainset.userId==uid] 
    else:
        dataset = df_target[df_target.userId==uid]
    for index, rows in dataset.iterrows():
        try: 
            prediction.append(opt_decomposistion.loc[int(rows['movieId']),int(rows['userId'])])    
            targets.append(rows['rating'])
        except KeyError:
            pass
    rmse  = RMSE(np.asarray(prediction), np.asarray(targets))
    return rmse 



if __name__ == "__main__":
    global df_target, prediction_cf, decomposistion, opt_decomposistion
    trainset, df_train = initialize_train_data()  
    df_target = initialize_test_data()
    # task1 initialize 
    similarity = cosine_similarity(df_train.fillna(0))
    # task 2 initialize
    decomposistion = build_svd(df_train) 
    # task 3
    opt_decomposistion = build_svd(df_train, 500)
    
    user_ids = read_user_id()
    result = do(user_ids)
    write_output(result)    


# In[ ]:




