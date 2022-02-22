import pandas as pd
import tweepy

# get user input for csv file path 
print('NOTE: The csv file must have precisely two columns: "Tweet ID" and "Label" (in that order).')
path = input('Enter file pathname: ')

# load DataFrame with tweet ID's and labels 
df = pd.read_csv(path)
df.columns = ['id', 'label']
print('File loaded! See sample below:')
print(df.head())

# get user input for Twitter API credentials
print('Now to access the Twitter API, please enter the following...')
public_key = input('Enter Consumer Public Key: ')
private_key = input('Enter Consumer Private Key: ')
public_token = input('Enter Public Access Token: ')
private_token = input('Enter Private Access Token: ')

# fire up the Twitter API using Tweepy 
auth = tweepy.OAuthHandler(public_key, private_key)
auth.set_access_token(public_token, private_token)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# since the API can only process 100 ID's at a time, the ID list is split into batches of 100 
id_batches = [list(df.id)[i:i+100] for i in range(0, len(list(df.id)), 100)]
tweet_texts = []

# loop through each ID batch 
for batch in id_batches:
    # get tweets from API
    batch_tweets = api.statuses_lookup(batch)
    # extract text and ID from each tweet and store in "tweet_texts"
    tweet_texts = tweet_texts + [{'id' : tweet.id, 'text' : tweet.text} for tweet in batch_tweets]

# merge "tweet_texts" with previous dataframe, joining on ID 
df_text = df.merge(pd.DataFrame(tweet_texts))
print('Finished with all API requests! See sample below:')
print(df_text.head())

# save new dataframe as csv 
print('NOTE: Filepath for new file must end in ".csv"')
save_path = input('Enter filepath for new file: ')
df_text.to_csv(save_path)

# closing message
print('All done!')