# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display

def read_user_id():
    with open('input.txt', 'r') as f:
        return [l.strip() for l in  f.readlines()]

def write_output(prediction):
    with open('output.txt', 'w') as f:
        for p in prediction:
            for r in p:
                f.write(r + "\n")

def do(ids):
    # TODO  
    # test implementation
    # prediction = [['{},{},{}'.format(i, 5, 3.5)]*30 for i in ids]
    prediction = [ get_top(uid) for uid in ids]
    return prediction


def get_top(uid, top=30):
    prediction = calcutate_rate(uid)
    prediction = np.around(prediction, 4)
    # display(prediction)

    prediction = prediction.sort_values().head(top).sort_index()
    display(prediction)
    return prediction.sort_values().head(top) 
    



def represent_movies():  
    global df_genre, representation
    df_genre = pd.read_csv('data/movies_w_imgurl.csv', usecols = ['movieId','genres'])
    
    # set of genre
    genres_list = []
    for line in df_genre['genres'].tolist():
        genres_list.extend(line.split('|'))
    genres_list = set(genres_list)
    genre_count = { genre:0 for genre in iter(genres_list)}  
    
    # count genre 
    for line in df_genre['genres'].tolist(): 
        items = line.split('|')
        for item in items:
            genre_count[item]+=1

    # check total count 
    movie_ids = set(df_genre['movieId'].tolist())
    total_count = len(movie_ids)
    
    representation = pd.DataFrame(movie_ids, columns=['movieId'])
    # display(pd.DataFrame(representation))

    # create IDF for genre_count 
    TF = 1
    for key in genre_count.keys():
        IDF = np.log10(total_count / genre_count[key])
        genre_count[key] = TF * IDF
    
    # fill representation 
    for genre in genres_list:
        representation[genre] = 0.0 

   
    for index,row in df_genre.iterrows(): 
       	items = row['genres'].split('|')
        for genre in genres_list:
            if genre in items:
               representation.at[index, genre] = genre_count[genre]
    # store(representation, "represent_genre.csv")
    # print(representation.shape)
            


def store(table, fname=None):
    if fname == None:
       fname = 'test.csv'
    table.to_csv(fname, mode='w')

       
def represent_tags():
    global tag_count, total_count
    df_tag = pd.read_csv('data/tags.csv', usecols=['userId','movieId','tag','timestamp'])
    
    tag_list = []
    for line in df_tag['tag'].tolist():
        tag_list.extend([ item.strip() for item in line.split(',')]) 
    tag_list = set(tag_list)
    tag_count = { tag: 0.0 for tag in iter(tag_list)}

    # init table for tag
    for key in tag_count.keys():
        representation[key]=0.0

    # calculate IDF for tag  
    ### count tags
    for row in df_tag['tag'].tolist():
        tags = [ item.strip() for item in row.split(',') ]
        for tag in tags:
            tag_count[tag]+=1

    ### check total count 
    movie_ids = set(df_tag['movieId'].tolist())
    total_count = len(movie_ids) 
             
    ### calculate IDF for tag  
    IDFs = dict()
    for key in tag_count.keys():
        IDFs[key] = np.log10(total_count / tag_count[key])
   
    # calacate TF for tag 
    ### read by movies by movieid  -> count total tags 
    ### there are cases of multiple user tagging a same movie, must consider the case of same tag used 
    ## movie id in tag.csv
    for movie_id in movie_ids:
        n_d = 0.0
    #### representation [ movie, tag] += 1 if exist
        try:
            mid_rows = df_tag[df_tag.movieId == movie_id]
            for index, row in mid_rows.iterrows():
                tags = [ item.strip() for item in row['tag'].split(',')]
                n_d += len(tags)
    ###### using df_tag [tag] as n_(t,d)
                for tag in tags:
                    representation.at[movie_id, tag] += 1
    ####### normalize tag TF with n_d
    ####### multipy IDF with TF
            if n_d == 0:
               continue
            for tag in tag_count.keys():
                representation.at[movie_id, tag] /= n_d
                representation.at[movie_id, tag] *= IDFs[tag]
        except: 
            pass
    
    # store(representation, 'representation_tag.csv')
    # print(representation.shape)
             

def cosine_sim(A, B):
    sim = dot(A, B) * 1.0 / (norm(A) * norm(B))  
    if math.isnan(sim):
       sim = 0.0
    return sim

# cosine_similarity( array A, array B)


def init_rate():
    global df_rate
    df_rate = pd.read_csv('data/ratings.csv', usecols=['userId','movieId','rating'])


def calcutate_rate(uid):
    # np.matmul(user_sim.T, user_rating) / (sim_sum + 1))
    rated_movieIds = df_rate[df_rate.userId == int(uid)]['movieId'].tolist()
    # user_sim = similarity[similarity['movieId'].isin(rated_movieIds)]
    user_sim = similarity.loc[rated_movieIds]
    # print('user_sim.T.to_numpy().shape : ' , user_sim.T.to_numpy().shape)
    # print('user_sim.to_numpy().shape : ', user_sim.to_numpy().shape)
    user_rating = df_rate[df_rate.userId == int(uid)]['rating']
    # print('user_rating.to_numpy().shape : ', user_rating.to_numpy().shape)
    # sim_sum = user_sim.T.to_numpy().sum()
    sim_sum = user_sim.sum()
    # print ('sim_sum.shape : ', sim_sum.shape)
    # np.matmul(user_sim.T, user_rating) / (sim_sum + 1)
    prediction = np.matmul(user_sim.T.to_numpy(), user_rating.to_numpy()) / (sim_sum + 1)
    return prediction
    
     
 
if __name__ == "__main__":
    # task 1
    represent_movies()
    # task 2
    represent_tags() 
    # task 3
    similarity = cosine_similarity(representation)
    movieIds = representation['movieId'].tolist()
    similarity = pd.DataFrame(similarity, columns=movieIds, index=movieIds)
    # store(similarity, "similarity.csv")
    
    # task 4 recommending
    init_rate()
    user_ids = read_user_id()
    result = do(user_ids)
    #write_output(result)
