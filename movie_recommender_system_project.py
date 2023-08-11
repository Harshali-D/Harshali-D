# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:32:47 2023

@author: harsahli
"""

import numpy as np
import pandas as pd 
import pickle


credits = pd.read_csv(r"C:\Users\harsahli\Desktop\tmdb_5000_credits.csv")
movies = pd.read_csv(r"C:\Users\harsahli\Desktop\tmdb_5000_movies.csv")

#both data sets(movie and credit)merged by the coloumn TITLE into a new datset called movies 

df = movies.merge(credits, on="title")
df=df[["movie_id","title","overview","genres","keywords","cast","crew"]]

#data cleaning 

df.isnull().sum()
df.dropna(inplace=True)

import ast

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L

df["genres"] = df["genres"].apply(convert)
df.head(1)

df["keywords"] = df["keywords"].apply(convert)


def convert2(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i["name"])
            counter+=1
        else:
            break
    return L

df["cast"] = df["cast"].apply(convert2)

def fetch(obj):
    L =[]
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L   

df["crew"] = df["crew"].apply(fetch)

df["overview"] = df["overview"].apply(lambda x:x.split())

# Removing " " (spaces) between Words from features
df["cast"] = df["cast"].apply(lambda x:[i.replace(" ","") for i in x])
df["crew"] = df["crew"].apply(lambda x:[i.replace(" ","") for i in x])
df["keywords"] = df["keywords"].apply(lambda x:[i.replace(" ","") for i in x])
df["genres"] = df["genres"].apply(lambda x:[i.replace(" ","") for i in x])


df["tags"] = df["overview"] + df["genres"] + df["keywords"] + df["cast"] + df["crew"]

#below is the new data set, with movie_id,title,tags

new_df = df[["movie_id", "title", "tags"]]

new_df["tags"] = new_df["tags"].apply(lambda x:" ".join(x))
new_df["tags"] = new_df["tags"].apply(lambda x:x.lower())

# Poreter Stemmer for handeling repeated words in tags feature
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)

new_df["tags"] = new_df["tags"].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words="english")

vector = cv.fit_transform(new_df["tags"]).toarray()


from sklearn.metrics.pairwise import cosine_similarity
similar = cosine_similarity(vector)

# Creating our Recommend function it will return Top 5 movies back
# Creating a function to recommend top 5 similar movies

def recommend(movie):
    movie_index = new_df[new_df["title"]==movie].index[0]
    distances = similar[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    print(f"you might also like-->")
    for i in movie_list: 
        print(new_df.iloc[i[0]].title)
        
user_input = input("Enter the name of the movie you like: ")
recommend(user_input)



pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))
pickle.dump(similar,open('similarity.pkl','wb'))





