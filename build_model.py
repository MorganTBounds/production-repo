# import required libraries 
import pandas as pd
import numpy as np
import os
import re
import contractions
import spacy
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from scipy.stats import uniform

# import spaCy English
nlp = spacy.load("en_core_web_sm")

############################################
###### Data Wrangling and Exploration ######
############################################

# import data
df1 = pd.read_csv(r'data/raw_data/hatespeech_id_label_text.csv', lineterminator='\n')
df2 = pd.read_csv(r'data/raw_data/labeled_data.csv', lineterminator='\n')
df3 = pd.read_csv(r'data/raw_data/NAACL_SRW_2016_text.csv', lineterminator='\n')
df4 = pd.read_csv(r'data/raw_data/train_E6oV3lV.csv', lineterminator='\n')

# drop everything but label and text
df1 = df1[['label', 'text']]

# change column names to 'label' and 'tweet'
df1.columns = ['label', 'tweet']

# change label to 1 if hate speech and 0 otherwise
df1.label = df1.label.apply(lambda x: 1 if x == 'hateful' else 0)

# drop everything but class and tweet 
df2 = df2[['class', 'tweet']]

# change column names to 'label' and 'tweet'
df2.columns = ['label', 'tweet']

# change label to 1 if hate speech and 0 otherwise
df2.label = df2.label.apply(lambda x: 1 if x == 0 else 0

# drop everything but label and text
df3 = df3[['label', 'text']]

# change column names to 'label' and 'tweet'
df3.columns = ['label', 'tweet']

# change label to 1 if hate speech and 0 otherwise
df3.label = df3.label.apply(lambda x: 0 if x == 'none' else 1)

# drop everything but label and tweet
df4 = df4[['label', 'tweet']]

# combine all four data frames into one
df = pd.concat([df1, df2, df3, df4])

# cast all labels as ints
df.label = df.label.astype(int)

# reset the index
df.reset_index(drop=True, inplace=True)

# drop duplicate tweets with contradictory labels
contradictory_tweets = df.groupby('tweet').label.nunique()[(df.groupby('tweet').label.nunique() > 1)].index
contradictory_index = df[df.tweet.isin(contradictory_tweets)].index
df.drop(index=contradictory_index, inplace=True)

def clean_tweet(tweet):
    # Use contractions package to split contractions
    tweet = contractions.fix(tweet.lower())
    # Use regex to remove special characters, numbers, mentions, emojis, websites, RT preambles
    tweet = " ".join(re.sub(r"(\brt\b)|(@[a-z0-9]+)|(http[\S]+)|([^a-z])", " ", tweet).split())
    # Use spaCy nlp to perform lemmatization
    doc = nlp(tweet)
    tweet = " ".join([token.lemma_ for token in doc if token.is_stop == False])
    return tweet 

df['clean_tweet'] = df.tweet.apply(clean_tweet)

df.dropna(inplace=True)

# Save the clean, wrangled data to csv 
df.to_csv('data/clean_data.csv', index=False)

#############################
###### Model Training ######
#############################

# import data
df = pd.read_csv('data/clean_data.csv', lineterminator='\n')

# Split data into features and target label 
X = df.clean_tweet
y = df.label

# Create preprocessing pipeline for data
pipe = Pipeline([('bow', CountVectorizer(min_df=2)),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])

# Create parameter space for randomized hyperparameter tuning         
distributions = {
    'bow__ngram_range' : [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf' : [True, False],
    'tfidf__norm' : ['l1', 'l2'],
    'model__solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'model__penalty' : ['l1', 'l2', 'none'],
    'model__C' : uniform(loc=0, scale=10),
    'model__max_iter': range(100, 1000),
    'model__class_weight' : ['balanced', None]
}

# Train model on all the data
clf = RandomizedSearchCV(pipe, distributions, cv=5, scoring='f1_macro', n_jobs=-1)
clf.fit(X, y);

# Pickle the model
pickle.dump(clf, open('model.pickle', 'wb'))






