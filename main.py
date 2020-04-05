# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv

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

GENRES = 3
def represent_movies():  
    global genre_count, total_count, movie_table
    with open('data/movies_w_imgurl.csv', 'r') as src:
         reader = csv.reader(src)
         # skip header 
         next(reader) 
         genre_count = dict()
         total_count= 0 
         for line in reader : 
             total_count+=1
             genres=line[GENRES].split('|')
             for genre in genres:
                 if genre_count.get(genre) == None:
                    genre_count[genre] = 0
                 genre_count[genre]+=1
         # create IDF for genre_count 
         TF = 1
         for key in genre_count:
             IDF = np.log10(total_count / genre_count[key])
             genre_count[key] = TF * IDF
         
     

def check_genre():
    with open('data/movies_w_imgurl.csv', 'r') as src:
         reader = csv.reader(src)
         # skip header 
         next(reader) 
         for line in reader : 
             genres=line[GENRES].split('|')
             out = "%s | "%line[0]
             for genre in genres:
                 out=out+"%s | "%genre_count[genre]  
             print(out) 

if __name__ == "__main__":
    # task 1
    represent_movies()
    check_genre()
 
    # task 4 
    user_ids = read_user_id()
    result = do(user_ids)
    write_output(result)
