"""WIP
filter out nbsp
Add other financial subreddits
Add other dimensions (r/news, r/worldnews, news, twitter)
Look at comments, especially in daily/weekly discussion
"""

import praw
import time
import pandas as pd
import time
from time import sleep
import numpy as np
import math
import csv

### NLP
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import re

import config
import tickers

start_time = time.perf_counter()

### INITIALIZE AND GATHER DATA

reddit = praw.Reddit(
    client_id="gMQhHTjivbo8gQ",
    client_secret="Wkai0BrDrcHtCdy4t_maPdl1agW6yA",
    password=config.PASSWORD,
    user_agent="Stock Extraction",
    username="clcironic"
)

subreddit = reddit.subreddit('wallstreetbets')
limit = 2000

def gather(subreddit, seconds_elapsed, to_file=None, post_flairs=['DD', 'Discussion', 'Gain', 'Loss']):
    posts_list = []
    for submission in subreddit:
        if submission.link_flair_text in post_flairs or not post_flairs:
            if not submission.stickied:
                if seconds_elapsed >= time.time()-submission.created_utc:
                    posts_list.append([submission.title, submission.selftext, submission.author.name, submission.author.comment_karma+submission.author.link_karma, submission.created_utc, time.time()-submission.created_utc, 
                        submission.link_flair_text, submission.id, submission.score, submission.upvote_ratio])

    df = pd.DataFrame(data=posts_list, columns=["Title", "Text", "Author", "Author Karma", "Date", "Seconds Ago", "Flair", "ID", "Upvotes", "Upvote Ratio"])
    return df

df = gather(subreddit.hot(limit=limit), 86400/2)

### SENTIMENT ANALYSIS

financial_df = pd.read_csv("financial_words.csv")
new_words = {}
def convert_words(df_series, val):
    for i in range(len(df_series)):
        new_words[str(df_series[i])] = val

convert_words(financial_df['negative'], -2)
convert_words(financial_df['positive'], 2)
convert_words(financial_df['uncertain'], 0)
new_words = {k.lower(): v for k, v in new_words.items()}

SIA = SentimentIntensityAnalyzer()
SIA.lexicon.update(new_words)

def sentiment(sentence):
    sentiment_dict = SIA.polarity_scores(sentence)
    return sentiment_dict['compound']

title_list = []
for title in df['Title']:
    title_list.append(sentiment(str(title)))

text_list = []
for text in df['Text']:
    text_list.append(sentiment(str(text)))

df['Title Sentiment'] = title_list
df['Text Sentiment'] = text_list
df['Overall Sentiment'] = np.array(title_list)+np.array(text_list)

### MATHEMATICAL MODELS / ADD OTHER FEATURES

df['Importance'] = np.log(df['Author Karma'])/np.log(10)*np.log(df['Upvotes']+1)/np.log(2)
df['Scaled Importance'] = round(df['Importance']/max(df['Importance'])*100, 3)

b = 0.7
th = 0.8

upvote_p_list = []
for i in df.index:
    upvote_p_list.append((1/np.power(0.2, b))*np.power(df.loc[i]['Upvote Ratio']-th, b) if df.loc[i]['Upvote Ratio'] >= th else (-1/np.power(0.2, b))*np.power(-df.loc[i]['Upvote Ratio']+th, b))
df['Upvote Polarity'] = upvote_p_list
df['Polarity'] = df['Upvote Polarity']*df['Overall Sentiment']
df.to_csv("all_data.csv")

### EMOJI EXTRACTION
def extract_emojis(s):
    s = str(s)
    emojis = ''
    for c in s:
        if ord(c) > 8500 and c not in emojis:
            emojis += c
    return emojis

### WORD EXTRACTION
stop_words = set(stopwords.words('english'))
table = str.maketrans('', '', string.punctuation)
lemmatizer = WordNetLemmatizer()

def clean(text):
    text = re.sub(r"www\S+", '', str(text), flags=re.MULTILINE) #filters out links
    text = re.sub(r"http\S+", '', str(text), flags=re.MULTILINE) #filters out links
    tokens = word_tokenize(text) #split into words
    tokens = [w.lower() if not w.isupper() else w for w in tokens] #make everything lowercase
    stripped = [w.translate(table) for w in tokens] #get rid of punctuation
    words = [word for word in stripped if word.isalpha()] #idk
    words = [w for w in words if not w in stop_words] #get rid of stop words like "no"
    lemma = [lemmatizer.lemmatize(word, 'v') for word in words] #lemmatization/stemming
    return lemma

counter_dict = {}
importance_dict = {}
polarity_dict = {}

avg_importance = 0
avg_polarity = 0
for i in df.index:
    importance = df.loc[i]['Scaled Importance']
    polarity = df.loc[i]['Polarity']
    labels_list = ['Title', 'Text']
    avg_importance += importance
    avg_polarity += polarity
    for label in labels_list:
        new_emojis = extract_emojis(df.loc[i][label])
        for c in new_emojis:
            try:
                counter_dict[c] += 1
                importance_dict[c] += importance
                polarity_dict[c] += polarity
            except:
                counter_dict[c] = 1
                importance_dict[c] = importance
                polarity_dict[c] = polarity

        clean_text = clean(df.loc[i][label])
        for word in clean_text:
            try:
                counter_dict[word] += 1
                importance_dict[word] += importance
                polarity_dict[word] += polarity
            except:
                counter_dict[word] = 1
                importance_dict[word] = importance
                polarity_dict[word] = polarity

### NORMALIZE
for word in counter_dict:
    importance_dict[word] /= counter_dict[word]
    polarity_dict[word] /= counter_dict[word]

avg_importance /= len(df)
avg_polarity /= len(df)

def sort_dict(tmp_dict):
    sorted_dict = dict(sorted(tmp_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict

### RETURN DATA ON TICKERS
ticker_labels = []
ticker_data = []
#tickers_list = {t.lower() for t in tickers.TICKERS_LIST}
for ticker in tickers.TICKERS_LIST:
    try:
        ticker_data.append([counter_dict[ticker], importance_dict[ticker], polarity_dict[ticker]])
        ticker_labels.append(ticker)
    except:
        pass

output_df = pd.DataFrame(data=ticker_data, index=ticker_labels, columns=["Count", "Avg Importance", "Avg Polarity"])
output_df = output_df.sort_values('Count', ascending=False)
output_df.to_csv("stocks.csv")

print(output_df)
print("This program scraped %d posts in %.3f seconds." % (len(df), time.perf_counter()-start_time))
print("This session had an average importance of %.3f and average polarity of %.3f." % (avg_importance, avg_polarity))