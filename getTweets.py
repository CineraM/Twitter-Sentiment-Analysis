import snscrape.modules.twitter as sntwitter

query = "(from:elonmusk exclude:replies)"
tweets = []
limit = 50

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.content])
        
print(tweets)