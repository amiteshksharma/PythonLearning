import requests
import os
import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
import re
warnings.filterwarnings('ignore')
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

##################################
#  Start of ML learning methods  #
##################################

data = {
    'Names': ['john','jay','dan','nathan','bob', 'giannis', 'jokic', 'lebron', 'durant'],
    'Politics': ['Trump', 'democrats', 'liberal', 'usps', 'progressive', 'president', 'congress', 'judicial',
    'court', 'supreme court', 'media', 'narrative'],
    'Countries': ['America', 'USA', 'US', 'China', 'India', 'Honduras', 'France', 'Germany'],
    'Sports': ['basketball', 'soccer', 'football', 'hockey', 'ball', 'tennis', 'racket'],
    'Vacation': ['world','flying','paradise','travel', 'America', 'trip', 'vacation'],
    'Gloomy': ['sad', 'struggling', 'depression', 'suicide', 'kms', 'hate', 'sucks'], 
}

Pronouns = ['i', "he", 'she', 'it', 'they', 'we', 'ours', 'you']  

categories = {word: key for key, words in data.items() for word in words}

# Load the whole embedding matrix
embeddings_index = {}
with open('glove.6B.100d.txt', encoding="utf8") as f:
  for line in f:
    values = line.split()
    word = values[0]
    embed = np.array(values[1:], dtype=np.float32)
    embeddings_index[word] = embed
print('Loaded %s word vectors.' % len(embeddings_index))
# Embeddings for available words
data_embeddings = {key: value for key, value in embeddings_index.items() if key in categories.keys()}

# Processing the query
def process(query):
    if query not in embeddings_index:
        return 'Value not in Embed'
    query_embed = embeddings_index[query]
    scores = {}
    for word, embed in data_embeddings.items():
        category = categories[word]
        dist = query_embed.dot(embed)
        dist /= len(data[category])
        scores[category] = scores.get(category, 0) + dist
    return scores

##################################
#  Start of Twitter API Methods  #
##################################

def auth():
    return os.environ.get("BEARER_TOKEN")

def create_url():
    query = "Amitesh2001"
    tweet_fields = 1
    url = "https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name={}&count={}&tweet_mode={}".format(query, tweet_fields, "extended")
    return url

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    print(response.status_code)
    if response.status_code != 200:
        print("There was an error")
    
    print(response)
    return response.json()

def get_response():
    bearer_token = auth()
    url = create_url()
    headers = create_headers(bearer_token)
    json_response = connect_to_endpoint(url, headers)
    return json_response

# Get the Responses from the Twitter API, which is the list of all my tweets
responses = get_response()

def create_dictionary():
    dictionary = {}
    for tweet in responses:
        string = tweet.get("full_text")
        split_string = string.split()
        for word in split_string:
            if word in dictionary:
                dictionary[word] = dictionary[word] + 1
            else:
                dictionary[word] = 1

    #Sort the dictionary to get the top words
    sort = sorted(dictionary.items(), key=lambda x:x[1], reverse=True)
    # print(sort)
    return sort

def tokenize_words():
    # Create the sentence to tokenize and tag
    words = ''
    for tweet in responses:
        words += tweet["full_text"] + " "
        
    # Tokenize and then tag the text with nltk
    tokenize = nltk.word_tokenize(words)
    tagging = nltk.pos_tag(tokenize)

    # Create the object to count the total number of adjs, nouns, etc
    total_number = {}
    # Loop through tagging to count each type of word
    for token in tagging:
        # If it is already in the dictionary, then increment by 1
        if token[1] in total_number:
            total_number[token[1]] = total_number[token[1]] + 1
        # Otherwise, start the count at 1
        else:
            total_number[token[1]] = 1
    print(total_number)

def categorize_words():
    word_dict = {}
    for tweet in responses:
        print(tweet.get('full_text'))
        word = tweet.get('full_text')
        arr_words = word.split()
        print(arr_words)
        for words in arr_words:
            if words.lower() in Pronouns:
                continue

            if "https" in words:
                continue
            
            regex = re.compile('[^a-zA-Z]')
            words = regex.sub('', words)

            print(words)
            process_word = process(words)
            print(process_word)
            
            max = 'Vacation'
            for categories in process_word:
                print(categories)
                
            

if __name__ == "__main__":
    create_dictionary()
    tokenize_words()
    categorize_words()