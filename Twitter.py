import requests
import os
import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from Scraper import scrape_prepositions, scrape_sports, scrape_politics, scrape_social_issues, scrape_science
import numpy as np
import warnings
import re
import operator
import mysql.connector
warnings.filterwarnings('ignore')
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

scrape_prepositions = []
scrape_sports = []
scrape_politics = []
scrape_social_issues = []
scrape_science = []
scrape_contractions = []
scrape_names = []
##################################
#  Start of MySQL Database call  #
##################################

pswd = os.environ.get('PASSWORD')
user = os.environ.get('USERNAME')
table = os.environ.get('TABLE')
host = os.environ.get('HOST')

cnx = mysql.connector.connect(user=user, password=pswd, database=table, host=host)
cursor = cnx.cursor()

query = "SELECT * FROM words"

cursor.execute(query)
result = cursor.fetchall()
for row in result:
    if row[0] != '':
        scrape_prepositions.append(row[0])

    if row[1] != '':
        scrape_sports.append(row[1])
    
    if row[2] != '':
        scrape_social_issues.append(row[2])

    if row[3] != '':
        scrape_politics.append(row[3])

    if row[4] != '':
        scrape_science.append(row[4])

    if row[5] != '':
        scrape_contractions.append(row[5])
    
    if row[6] != '':
        scrape_names.append(row[6])

##################################
#  Start of Twitter API Methods  #
##################################

def auth():
    return os.environ.get("BEARER_TOKEN")

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def create_url():
    query = "Amitesh2001"
    tweet_fields = 200
    url = "https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name={}&count={}&tweet_mode={}&trim_user={}".format(query, tweet_fields, "extended", True)
    return url

def create_url_trending():
    url = "https://api.twitter.com/1.1/trends/place.json?id=1"
    return url

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

def get_trending():
    bearer_token = auth()
    url = create_url_trending()
    headers = create_headers(bearer_token)
    json_response = connect_to_endpoint(url, headers)
    trending_array = []
    trends = json_response[0]['trends']
    for name in trends:
        try:
            name['name'].encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            continue
        else:
            trending_array.append(name['name'].lower())
    
    return trending_array

##################################
#  Start of ML learning methods  #
##################################

# 'Politics': ['senate', 'democrats', 'liberal', 'usps', 'progressive', 'president', 'congress', 'judicial',
#     'court', 'supreme court', 'media', 'narrative', 'election', 'conservatives', 'republican', 'legislation', 'representative', 'secretary'],

# Define categories to teach the machine how to categorize words
data = {
    'Names': scrape_names,
    'Politics': scrape_politics,
    'Athletics': scrape_sports,
    'Social Issues': scrape_social_issues,
    'Trendy': get_trending(),
    'Science': scrape_science,
}

Pronouns = ['i', "he", 'she', 'it', 'they', 'we', 'ours', 'you', 'a', 'an', 'the', 'im', "i'm", 'its', 'their', "it's", 'them', 'me', 'him', 'her', 'this', 'that']  
Acronyms = ['smh', 'lol', 'lmfao', 'ttyl', 'gtfoh', 'stfu', 'rofl', 'lmk', 'ily', 'nvm', 'pov', 'rip', 'ppl', 'rt', 'etc', 'jfc', 'yo', 'ngl']
Verbs = ['are', 'were', 'will', 'is', 'was', 'have', 'has', 'had', 'is', 'am', 'do', 'does', 'can']
Questions = ['who', 'what', 'when', 'where', 'why', 'how']
Conjunctions = ['for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'if', 'no', 'not', 'be', 'while', 'then', 'next', 'first', 'second']
Alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
Possessive = ['my', 'our', 'your', 'his', 'her', 'its', 'their', 'mine', 'ours', 'yours', 'his', 'hers', 'its', 'theirs', 'ur']
Expletives = ['damn', 'fuck', 'fck', 'shit', 'crap', 'bitch', 'bish', 'bastard', 'dumb', 'dumbest', 'ass', 'fucking', 'fucked']
Prefixes = ['dr', 'mr', 'ms', 'mrs', 'miss', 'mister', 'doctor', 'proff', 'prof', 'professor', 'bro', 'dawg', 'man', 'girl', 'son', 'lady', 'boy', 'dude']

Prepositions = scrape_prepositions
Contractions = scrape_contractions

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
        return '404'
    query_embed = embeddings_index[query]
    scores = {}
    for word, embed in data_embeddings.items():
        category = categories[word]
        dist = query_embed.dot(embed)
        dist /= len(data[category])
        scores[category] = scores.get(category, 0) + dist
    return scores

##################################
#  Start of ML learning process  #
##################################

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
    # print(total_number)

def categorize_words():
    word_dict = {}
    data = {
        'Names': [],
        'Politics': [],
        'Athletics': [],
        'Social Issues': [],
        'Trendy': [],
        'Science': [],
    }

    for tweet in responses:
        word = tweet.get('full_text')
        sort_word_into_category(word, word_dict, data)

        if "quoted_status" in tweet:
            retweet = tweet.get('quoted_status')['full_text']
            sort_word_into_category(retweet, word_dict, data)
 
    print(data)
    text_file = open("Output.txt", "w")
    text_file.write("Names: %s \n" % data.get('Names'))
    text_file.write("Politics: %s \n" % data.get('Politics'))
    text_file.write("Athletics: %s \n" % data.get('Athletics'))
    text_file.write("Social Issues: %s \n" % data.get('Social Issues'))
    text_file.write("Trendy: %s \n" % data.get('Trendy'))
    text_file.write("Science: %s \n" % data.get('Science'))
    text_file.close()
    return word_dict

def sort_word_into_category(word, word_dict, data):
    arr_words = word.split()
    for words in arr_words:
        words = words.lower()
        regex = re.compile('[^a-zA-Z]')
        words = regex.sub('', words)

        if words in Pronouns or words in Prepositions or \
        words in Acronyms or words in Verbs or "https" in words \
        or words in Questions or words in Conjunctions or words in Alphabet \
        or words in Possessive or words in Expletives or words in Contractions \
        or words in Prefixes:
            continue

        process_word = process(words)
        
        if process_word == '404':
            continue
        else:
            largest_value = max(process_word.items(), key=operator.itemgetter(1))[0]

            if largest_value in word_dict:
                word_dict[largest_value] = word_dict[largest_value] + 1
            else:
                word_dict[largest_value] = 1
            
            if words not in data[largest_value]:
                data[largest_value].append(words)
            

if __name__ == "__main__":
    # create_dictionary()
    # tokenize_words()
    print(categorize_words())