# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 20:12:07 2020

@author: pjb50
"""
import streamlit as st
import os
import pandas as pd

from gensim.test.utils import datapath
from gensim import utils
import gensim.models
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import nltk
import collections
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.corpus import stopwords
from nltk.collocations import *
import itertools
from surprise import Reader, Dataset
import pickle
import json
from surprise import KNNBasic

import umap
from bokeh.plotting import figure
from PIL import Image
import regex as re
import webbrowser
import random


img=Image.open('data/flav.png')
st.image(img, use_column_width=True)

st.title("Flavor pairing recommendation system")
with open('data/word2vec1m.pkl', 'rb') as f:
    model = pickle.load(f)
with open('data/knnBASIC_pred2.pkl', 'rb',) as f:
    predictions = pickle.load(f)
dfflav=pd.read_csv('data/falvor.tsv',sep='\t')
dfpairs=pd.read_csv('data/final_score') 
hover_data=pd.read_csv('data/umaphover.csv')

UMAP = pickle.load((open('data/UMAPwrv.sav', 'rb')))
x = st.text_input("Input an ingredient to see compatible ingredients:  ", "garlic")
# Add a selectbox to the sidebar:

def flavour_bible_look_up(x):
    in_df = dfflav['main'] == x
    resultdf=dfflav['pairing'].loc[in_df]
    w=resultdf[8:28]
    search_term=w.tolist()
    url = "https://www.google.com.tr/search?q={}".format(x +" "+str(search_term)+" "+"recipe")    
    z=webbrowser.open(url)
    return st.write(w), z


def word2vec_recipe1m(x):
    result=model.wv.most_similar(x)
    z=pd.DataFrame(result, columns=['pairs','score'])
    search_term=z['pairs'].tolist()
    url = "https://www.google.com.tr/search?q={}".format(x +" "+str(search_term)+" "+"recipe")    
    h=webbrowser.open(url)
    return z['pairs']
    st.write(z['pairs'])


def knn_recs(x):
    result= predictions[x]
    return st.write("KNN SURPRISE", result)

def paireddf_look_up(x):
    in_df = dfpairs['ingr1'] == x
    resultdf=dfpairs[['ingr2']].loc[in_df]
    resultdf=resultdf.rename(columns={'ingr2':'pairs'})
    search_term=random.sample(resultdf['pairs'].tolist(), 5)
    
    return st.write( resultdf[:10])

def accept_user_data():
    user_input= st.text_input("Enter an ingredient (lowercase): ")
    return user_input(kwargs)
    
def compare_all(x):
    a = word2vec_recipe1m(x)
    b = paireddf_look_up(x)
    c = knn_recs(x)
    d = flavour_bible_look_up(x)
    return st.write(a, b , c, d)

import plotly 
import plotly.graph_objs as go

st.write("""
          Recommended flavor pairs, take these and create a new dish!
""")
st.write(paireddf_look_up(x))
st.write(""" 
         ## WORD2VEC EMBEDDING OF INGREDIENTS

 ZOOM IN ON REGIONS TO EXPLORE
""")
st.write(""" **Check out our Tableau dash [link](https://public.tableau.com/profile/patrick.joseph.broderick#!/vizhome/FLAVORMAP/Dashboard1)** """)

fig = go.Figure()
trace = go.Scatter(x=UMAP.embedding_[:, 0], y=UMAP.embedding_[:, 1], mode='text',  text=hover_data['label'])
fig.add_trace(trace)
fig.update_layout(
autosize=True,
width=2000,
height=1000)
st.plotly_chart(fig)




#p=umap.plot.interactive(UMAP, point_size=8,labels=hover_data['label'],hover_data=hover_data, width=2000, height=2000)









print("run")

#st.line_chart(hover_data.cluster)