# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv
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
    global genre_count, total_count, representation, movie_ids
    df = pd.read_csv('data/movies_w_imgurl.csv', usecols=['movieId','genres'])
    # set of genre
    genres_list = []
    for line in df['genres'].tolist():
        genres_list.extend(line.split('|'))
    genre_count = { genre:0 for genre in iter(genres_list)} 
    
    for line in df['genres'].tolist(): 
        items = line.split('|')
        for item in items:
            genre_count[item]+=1
    
    # check total count 
    movie_ids = df.drop_duplicates(['movieId']).tolist()
    total_count = len(movie_ids)
    print ("movie_count : "+str(total_count))
    # create IDF for genre_count 
    TF = 1
    for key in genre_count.keys():
        IDF = np.log10(total_count / genre_count[key])
        genre_count[key] = TF * IDF
    
    # fill representation 
    representation = df
    for genre in genres_list:
        representation[genre] = df.apply(lambda x : 0, axis=1) 

    for line in representation:
        print(line)
        items = line['genre'].split('|')
        for genre in genres_list:
            if genre in items:
               line[genre] = genre_count[genre]
            else:
               line[genre] = 0 
            
    print ("genre shape: %d" %(representation.shape))
        
        
    

def represent_tags():
    global tag_count, total_count
    df = pd.read_csv('data/tags.csv', usecols=['userId','movieId','tag','timestamp'])
    tag_count = dict()
    # count tags
    for line in df['tag'].tolist():
        tags = line.split(',').strip()
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
