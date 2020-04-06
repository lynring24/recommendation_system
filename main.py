# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#from unittest.py import *

def read_user_id():
    with open('input.txt', 'r') as f:
        return [l.strip() for l in  f.readlines()]


def write_output(prediction):
    with open('output.txt', 'w') as f:
        for p in prediction:
            for r in p:
                f.write(r + "\n")

def do(ids):
    # test implementation
    prediction = [['{},{},{}'.format(i, 5, 3.5)]*30 for i in ids]
    return prediction

def represent_movies():  
    global genre_count, total_count, movie_ids
    df = pd.read_csv('data/movies_w_imgurl.csv', usecols = ['movieId','genres'])
    #df = pd.read_csv('data/movies_w_imgurl.csv', names = ['movieId', 'id', 'movieName' ,'genres', 'URL'])
    # set of genre
    genres_list = []
    for line in df['genres'].tolist():
        genres_list.extend(line.split('|'))
    genres_list = set(genres_list)
    genre_count = { genre:0 for genre in iter(genres_list)}  
    
 
    for line in df['genres'].tolist(): 
        items = line.split('|')
        for item in items:
            genre_count[item]+=1
    # check total count 
    movie_ids = set(df['movieId'].tolist())
    total_count = len(movie_ids)
    print ("movie_count : "+str(total_count))
    # create IDF for genre_count 
    TF = 1
    for key in genre_count.keys():
        IDF = np.log10(total_count / genre_count[key])
        genre_count[key] = TF * IDF
        print (key, genre_count[key])
    
    # fill representation 
    for genre in genres_list:
        df[genre] = 0.0 

    print (df.shape)
    for index,row in df.iterrows(): 
       	items = row['genres'].split('|')
        for genre in genres_list:
            if genre in items:
               df.at[index, genre] = genre_count[genre]
    store(df)
            


def store(table, fname=None):
    if fname == None:
       fname = 'test.csv'
    table.to_csv(fname, mode='w')



       
def represent_tags():
    global tag_count, total_count
    df = pd.read_csv('data/tags.csv', usecols=['userId','movieId','tag','timestamp'])
    tag_count = dict()
    # count tags
    for index, row in df['tag'].tolist():
        tags = row.split(',').strip()
        for tag in tags:
            if tag_count.get(tag) == None:
               tag_count[tag] = 0
            tag_count[tag]+=1
    # check tag 
    print ("total tag : "+str(len(tag_count.keys())))
    # calculate IDF for tag_count  
    IDFs = dict()
    for key in tag_count.keys():
        IDFs[key] = np.log10(total_count / tag_count[key])

             

if __name__ == "__main__":
    # task 1
    represent_movies()
    #check_genre()
    # task 2
    
 
    # task 4 
    user_ids = read_user_id()
    result = do(user_ids)
    write_output(result)
