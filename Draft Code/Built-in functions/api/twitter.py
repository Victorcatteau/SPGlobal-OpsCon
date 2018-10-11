#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
TWITTER
Started on the 25/04/2017

theo.alves.da.costa@gmail.com
theo.alvesdacosta@ekimetrics.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



# USUAL LIBRARY
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import sys
from bson.son import SON #to use sorted dictionaries
import re

# TWEEPY
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

# EKIMETRICS CUSTOM LIBRARY
from ekimetrics.nlp.models import *
from ekimetrics.api.utils import parse
from ekimetrics.utils import io
from ekimetrics.visualizations import charts as viz


# PYMONGO
try:
    from pymongo import MongoClient
except:
    pass


# PLOTLY
try:
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.graph_objs as go
except:
    pass


#=============================================================================================================================
# TOKENS & AUTHENTIFICATION
#=============================================================================================================================





# Access Tokens & API

consumer_key = 'rWDKjIcHmII334ht10Fn6MaOc'
consumer_secret = 'W3IOrszJX9m1Y3z2cXb5d08UaCsT02TlyJI8MidKz8vLKH2lFm'
access_token = '4090360696-Wn2ZldY8cnbb98EIZ0w0t76JLC3HTVOh3KJEGbY'
access_token_secret = 'TuSnK10mYRpP2bv7uBiE5Bp6lX9HbqAmWXJNdJFirJci5'




default_twitter_vocabulary = ["RT","&amp"]
twitter_language_mapping = {
    "en":"english"
}


def map_twitter_language(language):
    if language in twitter_language_mapping:
        return twitter_language_mapping[language]
    else:
        return language








#=============================================================================================================================
# TWITTER STREAMER
#=============================================================================================================================



class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)




class Twitter_Streamer(object):
    """
    Twitter streamer is a class to stream a Twitter feed on the keywords
    It returns a txt file with all the tweets metadata in it. 
    """


    def __init__(self,keyword = None):
        self.keyword = keyword
        self.stream = None
        self.is_streaming = False



    def streaming(self,keyword = None,path = "twitter_data/",async = True):
        """
        Stream launches the streaming of Twitter according to the keywords, 
        It opens a file and streams the data into it
        """

        # KEYWORD CHOICE
        print(keyword)
        if self.stream is not None:
            self.disconnect()

        keyword = [keyword] if type(keyword) != list else keyword
        self.is_streaming = True

        print(">> Launching streaming for keyword : {}".format(keyword))

        # SET THE STDOUT SYSTEM
        self.orig_sys = sys.stdout

        # CREATE THE FOLDER IF NOT EXISTING
        if not os.path.exists(path):
            os.mkdir(path)

        # Set the out print to the out file, so that each print is made in the file
        sys.stdout = open(path+"_".join(keyword)+'.txt','w',encoding = 'utf-8')

        # Prepare the listener
        l = StdOutListener()
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        # Launch the streamer
        self.stream = Stream(auth, l)
        self.stream.filter(track=keyword,async = async)


    def disconnect(self):
        sys.stdout = self.orig_sys
        self.stream.disconnect()
        self.is_streaming = False




















#=============================================================================================================================
# TWEETS ONTOLOGY
#=============================================================================================================================


class Tweets(Corpus):
    """
    Tweets is an abstraction class to study the content and the properties of a ensemble of Tweets. 
    It has many properties inherited from the base NLP class Corpus 
    """

    def __init__(self,json_path = None,json_data = None,verbose = 1):

        self.verbose = verbose

        # LOADING DATA
        if json_path is not None:
            tweets = self.load_data_from_json_file(json_path)
        elif json_data is not None:
            tweets = self.load_data_from_json(json_data)
        else:
            raise ValueError("You have to provide data")



        # CREATION OF THE TWEETS CORPUS
        super().__init__(documents = tweets,verbose = verbose)


    def load_data_from_json_file(self,json_path):
        json_data = io.open_json_data(json_path)
        tweets = self.load_data_from_json(json_data)
        return tweets



    def load_data_from_json(self,json_data):

        tweets = []
        empty_tweets = 0

        for i,tweet_data in enumerate(json_data):
            self.print("\r[{}/{}] Parsing and analyzing tweet data".format(i+1,len(json_data)),end = "")
            tweet = Tweet(json_data = tweet_data)
            if not tweet.empty_tweet:
                tweets.append(tweet)
            else:
                empty_tweets += 1

        if empty_tweets > 0:
            self.print(" - skipped {} empty tweets".format(empty_tweets))

        self.print("")

        return tweets


    def save_as_json(self,json_path):
        json_data = [tweet.meta_data for tweet in self.tweets]
        io.save_data_as_json(json_data,json_path)


    #----------------------------------------------------------------------------------
    # GETTERS

    def get_hashtags(self,with_performances = False):
        return [tweet.hashtags for tweet in self.documents]


    def get_users(self):
        pass


    def get_entities(self):
        pass


    def create_user_corpus(self):
        pass



    #----------------------------------------------------------------------------------
    # ANALYSIS


    def show_hashtags_distribution(self,with_plotly = True,on_notebook = True):
        all_hashtags = [h for hashtags in self.get_hashtags() for h in hashtags]
        all_hashtags = pd.DataFrame(all_hashtags,columns = ["hashtag"])
        all_hashtags["count"] = 1
        all_hashtags = all_hashtags.groupby("hashtag").sum().sort_values("count",ascending = False)

        viz.plot_simple_bar_chart(all_hashtags,with_plotly = with_plotly,on_notebook = on_notebook)




    def show_language_distribution(self,on_notebook = True):
        pass


    def show_geo_distribution(self,on_notebook = True):
        pass







    def show_users_distribution(self):
        pass


    def show_entities_distribution(self):
        pass




    #----------------------------------------------------------------------------------
    # HASHTAGS COOCURRENCES

    def build_hashtags_similarity_network(self):
        pass










#=============================================================================================================================
# TWEET ONTOLOGY
#=============================================================================================================================

BOT_VOCABULARY = [
    "makeyourownlane",
    "defstar5",
]





class Tweet(Document):
    """
    Tweet is an abstraction class to study the content and the properties of a Tweet. 
    It has many properties inherited from the base NLP class Document 
    """
    def __init__(self,json_data = None):
        
        # JSON PARSING
        if json_data is not None:
            self.meta_data = json_data
            
            parsing = self.parsing(json_data)

            # Protection for empty tweets
            if self.empty_tweet:
                return None


        # CLEANSING
        self.clean_tweet()
        self.lower()
        self.remove_vocabulary(BOT_VOCABULARY)

        # DOCUMENT INITIALIZATION
        super().__init__(text = self.text,title = self.raw_text,default_language = self.language)
        self.filter_punctuation()
        self.filter_on_length(min_length = 1)



    #----------------------------------------------------------------------------------
    # REPRESENTATION

    def __repr__(self):
        return self.raw_text


    def __str__(self):
        return self.__repr__()



    #----------------------------------------------------------------------------------
    # PARSING


    def parsing(self,json_data = None):
        """
        Function to parse the tweet full json data coming from the API extraction
        Affects several properties and attributes to the Tweet ontology
        """

        if json_data is None:
            json_data = self.meta_data


        if "text" not in json_data:
            self.raw_text = "Empty tweet"
            self.empty_tweet = True
            return None
        else:
            self.empty_tweet = False


        #-------------------------------------------------------------
        # SIMPLE PARSING

        # TEXT
        self.raw_text = parse(json_data,"text")

        # LANGUAGE
        self.language = map_twitter_language(parse(json_data,"lang"))

        # ID
        self.id = parse(json_data,"id")

        # GEO
        self.geo = parse(json_data,"geo")

        # CREATION DATE
        self.created_at = pd.to_datetime(parse(json_data,"created_at"))


        #-------------------------------------------------------------
        # COMPLEX PARSING

        # HASHTAGS
        self.hashtags = parse(json_data,["entities","hashtags","text"],none_value = [])

        # MENTIONS
        self.mentions = parse(json_data,["entities","user_mentions",["id","name","screen_name"]],none_value = [])

        # MENTIONS
        self.urls = parse(json_data,["entities","urls",["url","expanded_url"]],none_value = [])

        # KPIs
        self.kpis = {
            "retweets":parse(json_data,"retweet_count",none_value = 0),
            "likes":parse(json_data,"favorite_count",none_value = 0),
        }


        #-------------------------------------------------------------
        # RETWEET PARSING

        # PARSE THE RETWEET DATA AS ITS OWN TWEET DATA
        retweet_data = parse(json_data,"retweeted_status")

        # IF THERE IS NO DATA
        if retweet_data is None:
            self.is_retweet = False
            self.retweet = None

        # IF IT IS A RETWEET
        else:
            self.is_retweet = True

            # Create a tweet ontology for the original tweet
            self.retweet = Tweet(json_data = retweet_data)
            
            # Add the urls from the meta data if it's not contained in the main urls (can be cut on a retweet)
            urls = parse(retweet_data,["entities","urls",["url","expanded_url"]])
            for url in urls:
                if url not in self.urls:
                    self.urls.append(url)




        #-------------------------------------------------------------
        # USER PARSING

        # USER
        self.user = User(json_data = parse(json_data,"user"))




    #----------------------------------------------------------------------------------
    # CLEANSING

    def clean_tweet(self):

        def clean(t):
            return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", t).split())

        self.text = clean(self.raw_text)



    def lower(self):
        self.text = self.text.lower()




    def remove_vocabulary(self,vocabulary = []):
        vocabulary = vocabulary + default_twitter_vocabulary 
        for word in vocabulary:
            self.text = self.text.replace(word.lower(),"")


























#=============================================================================================================================
# USER ONTOLOGY
#=============================================================================================================================



class User(object):
    """
    User is an abstraction class to study an user on Twitter
    """
    def __init__(self,json_data = None):

        # JSON PARSING
        if json_data is not None:
            self.meta_data = json_data
            self.parsing(json_data)


    #----------------------------------------------------------------------------------
    # REPRESENTATION

    def __repr__(self):
        string = "-- {} --\n".format(self.name)
        string += self.description
        return string


    def __str__(self):
        return self.__repr__()




    #----------------------------------------------------------------------------------
    # PARSING

    def parsing(self,json_data = None):
        """
        Function to parse the tweet full json data coming from the API extraction
        Affects several properties and attributes to the Tweet ontology
        """

        if json_data is None:
            json_data = self.meta_data


        #-------------------------------------------------------------
        # SIMPLE PARSING

        # LANGUAGE
        self.language = parse(json_data,"lang")

        # ID
        self.id = parse(json_data,"id")

        # NAME
        self.name = parse(json_data,"name")
        self.screen_name = parse(json_data,"screen_name")

        # URL
        self.url = parse(json_data,"url")

        # LOCATION
        self.location = parse(json_data,"location")

        # DESCRIPTION
        self.description = parse(json_data,"description")

        # CREATION DATE
        self.created_at = pd.to_datetime(parse(json_data,"created_at"))



        #-------------------------------------------------------------
        # COMPLEX PARSING


        self.pictures = {
            "profile_banner": parse(json_data,"profile_banner_url"),
            "profile_image": parse(json_data,"profile_image_url_https"),
        }


        self.kpis = {
            "favorites": parse(json_data,"favourites_count",none_value = 0) + parse(json_data,"favorites_count",none_value = 0),
            "statuses": parse(json_data,"statuses_count",none_value = 0),
            "friends": parse(json_data,"friends_count",none_value = 0),
            "followers": parse(json_data,"followers_count",none_value = 0),
        }

        








#=============================================================================================================================
# MONGO TWEETS ONTOLOGY
#=============================================================================================================================




class MongoTweets(object):
    """
    MongoTweets is a class used to be connected on a MongoDB database storing the direct outputs of the Twitter API.
    MongoTweets can be connected with the Tweets ontology to do a NLP analysis of the content of the Tweets.
    Yet it has many capabilities such as analyzing the distribution of hashtags or languages even in a Big Data settings thanks to NoSQL aggregations
    """
    
    def __init__(self,database,collection,host,port = 27017,verbose = 1):

        # Initialization
        self.verbose = verbose
        self.collection = MongoClient(host,port)[database][collection]
        self.print(">> Connected successfully to the mongodb collection {} in {}".format(collection,database))


    def print(self,message):
        if self.verbose: print(message)


    #-------------------------------------------------------------
    # TWEETS PARSER


    def build_tweets_corpus(self,n = 5,query = {},hashtags = None):
        if hashtags is not None:
            if type(hashtags)!=list: hashtags = [hashtags]
            hashtags = [h.replace("#","").lower() for h in hashtags]
            query = {"entities.hashtags.text":{"$in":hashtags}}

        json_data = self.get_data(n = n,query = query)
        corpus = Tweets(json_data = json_data)
        return corpus




    def build_users_corpus(self,n = 5):
        user_data = self.get_users(n = n)
        texts = []
        titles = []
        for user in user_data:
            texts.append(user["description"])
            titles.append(user["name"])

        corpus = Corpus(texts = texts,titles = titles)
        return corpus






    #----------------------------------------------------------------------------------
    # GETTERS

    def get_texts(self,query = {},n = 100):
        data = []

        for tweet in tqdm(self.collection.find(query,limit = n)):
            data.append(tweet["text"])

        return data


    def get_data(self,query = {},n = 100):
        data = []

        for tweet in tqdm(self.collection.find(query,limit = n)):
            data.append(tweet)


        return data




    def get_users(self,query = {},n = 100):
        data = []

        for tweet in tqdm(self.collection.find(query)):
            user_data = {k:v for k,v in tweet["user"].items() if k in ["description","name"]}

            if user_data["description"] is None:
                continue
                
            user_data["description"] = user_data["description"].replace("#","").replace("@","")

            if user_data not in data:
                data.append(user_data)
                i += 1

            if i == n:
                break

        return data





    def get_hashtags(self,with_performances = False):
        pass





    def get_entities(self):
        pass



    #----------------------------------------------------------------------------------
    # UTILS

    def aggregate(self,pipeline):
        return pd.DataFrame(list(self.collection.aggregate(pipeline))).set_index("_id")


    def filter(self):
        pass



    def show_distribution(self,pipeline,display = True,with_plotly = True,on_notebook = True,top = 30):
        # QUERY
        data = self.aggregate(pipeline)

        # DISPLAYING
        if display:
            data = data.iloc[:top]
            fig = viz.plot_simple_bar_chart(data,with_plotly,on_notebook)
            return fig

        # RETURNING RESULTS
        else:
            data["%"] = data["count"]/data["count"].sum()
            data = data.iloc[:top]
            return data




    #----------------------------------------------------------------------------------
    # ANALYSIS



    def count(self):
        return self.collection.count()



    def show_language_distribution(self,display = True,with_plotly = True,on_notebook = True,top = 30):
        # MONGODB PIPELINE
        pipeline = [
            {"$group": {"_id": "$lang", "count": {"$sum": 1}}},
            {"$sort": SON([("count", -1), ("_id", -1)])}
        ]

        return self.show_distribution(pipeline,display,with_plotly,on_notebook,top)






    def show_hashtags_distribution(self,display = True,with_plotly = True,on_notebook = True,top = 30,with_performances = False):
        # MONGODB PIPELINE
        pipeline = [
            {"$unwind": "$entities.hashtags"},
            {"$group": {"_id": { "$toLower": "$entities.hashtags.text"}, "count": {"$sum": 1}}},
            {"$sort": SON([("count", -1), ("_id", -1)])}
        ]

        return self.show_distribution(pipeline,display,with_plotly,on_notebook,top)





    def show_geo_distribution(self,display = True,with_plotly = True,on_notebook = True,top = 30,by_country = True):

        # GEO LEVEL SELECTION
        if by_country:
            geo_level = "place.country"
        else:
            geo_level = "place.name"


        # MONGODB PIPELINE
        pipeline = [
            {"$match": {geo_level: { "$exists": True, "$ne": None }}},
            {"$group": {"_id": { "$toUpper": "$"+geo_level}, "count": {"$sum": 1}}},
            {"$sort": SON([("count", -1), ("_id", -1)])}
        ]

        return self.show_distribution(pipeline,display,with_plotly,on_notebook,top)








    def show_users_distribution(self,display = True,with_plotly = True,on_notebook = True,top = 30):
        # MONGODB PIPELINE
        pipeline = [
            {"$match": {"user.name": { "$exists": True, "$ne": None }}},
            {"$group": {"_id": "$user.name", "count": {"$sum": 1}}},
            {"$sort": SON([("count", -1), ("_id", -1)])}
        ]

        return self.show_distribution(pipeline,display,with_plotly,on_notebook,top)






    def show_entities_distribution(self,display = True,with_plotly = True,on_notebook = True,top = 30):
        pipeline = [
            {"$unwind": "$entities.user_mentions"},
            {"$group": {"_id": "$entities.user_mentions.name", "count": {"$sum": 1}}},
            {"$sort": SON([("count", -1), ("_id", -1)])}
        ]

        return self.show_distribution(pipeline,display,with_plotly,on_notebook,top)


















#=============================================================================================================================
# DEPRECATED
#=============================================================================================================================






#=============================================================================================================================
# TWITTER LOADER
#=============================================================================================================================



class TwitterTextLoader(object):
    def __init__(self,file_path = None,json_data = None,title = "twitter data"):

        # LOAD DATA
        if file_path is not None:
            self.data = self.load_file_data(file_path)
            self.title = file_path.split("/")[-1].replace(".txt","")
        elif json_data is not None:
            self.data = json_data
            self.title = title
        else:
            raise ValueError("You have to provide data to parse")





    #----------------------------------------------------------------------------------
    # REPRESENTATION

    def __repr__(self):
        return "{} : {} tweets".format(self.title,len(self.data))


    def __str__(self):
        return self.__repr__()




    #----------------------------------------------------------------------------------
    # IO

    def load_file_data(self,file_path):
        data = []
        tweets_file = open(file_path, "r").read().split('\n')
        for line in tweets_file:
            try:
                tweet = json.loads(line)
                data.append(tweet)
            except:
                continue
        return data