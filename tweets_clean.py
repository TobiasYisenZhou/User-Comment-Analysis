# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:17:10 2022

@author: zsk98
"""

import pandas as pd
import numpy as np
import re
import nltk

# read in data
df=pd.read_excel(r"D:\CU\spring 2022\viz\twitter\2021_raw_tweets.xlsx")

# defining functions and preparing dictionary
nltk.download('words')
words = set(nltk.corpus.words.words())

def english_keeper(var):
    words = set(nltk.corpus.words.words())
    return " ".join(w for w in nltk.wordpunct_tokenize(var) if w.lower() in words or not w.isalpha())

def clean_txt(txt_in):
    import re
    clean_str = re.sub("[^A-Za-z]+", " ", str(txt_in)).strip().lower()
    return clean_str

def word_count(var_i):
    tmp = len(var_i.split())
    return tmp 

def my_stop_words(var_in, new=[]):
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    for i in new:
        sw.append(i)
    tmp = [word for word in var_in.split() if word not in sw]
    tmp = ' '.join(tmp)
    return tmp

def my_stem(var_in):
    from nltk.stem.porter import PorterStemmer
    my_stem = PorterStemmer()
    tmp = [my_stem.stem(word) for word in var_in.split()]
    tmp = ' '.join(tmp)
    return tmp


df['cleaned']=df["text"].apply(clean_txt)       # dropping special characters or letter in other language
df["english"]=df["cleaned"].apply(english_keeper) # keep english words
df["len"]=df["english"].apply(word_count)         # calculate length for the english column
df1=df[df["len"]>5]                              # Keep cleaned tweets have more than 15 words (or other number)

 
new_sw=["play","game","a","b","m","f","d","c","e","g","h","i","j","k","l","n","o","p","q","r",
        "s","t","u","v","w","x","y","z"]                           # what other stop words we want to drop
df1["sw_removed"]=df1["english"].apply(lambda x: my_stop_words(x,new=new_sw)) #drop stop words


df1["stemmed"]=df1["sw_removed"].apply(my_stem)   # stemming


df1.to_csv(r"D:\CU\spring 2022\viz\twitter\2021_cleaned_tweets.csv")
