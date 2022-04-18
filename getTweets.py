import snscrape.modules.twitter as sntwitter
import re

def remove_url(text): 
    url_pattern  = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub(r'', text)

def query_tweet(url):
    # tweet example
    # "https://twitter.com/elonmusk/status/1514681422212128770?cxt=HHwWhICz7c6unYUqAAAA"
    # get ridd off the last fslash -> '/'
    # tweet_id will be --> 1514681422212128770?...
    last_forwardslash = url.rindex('/')
    tweet_id = url[last_forwardslash+1:]

    #  a tweet id only has digits --> remove ?cxt=HHwWh...
    lastdigit = 0
    for i in range(len(tweet_id)):
        if not tweet_id[i].isnumeric():
            lastdigit = i
            break

    if lastdigit > 0:
        tweet_id = tweet_id[:lastdigit]

    tweet = ""
    # loop that gets content from a tweet --> user, tweet, followers...
    for item in sntwitter.TwitterTweetScraper(tweetId=tweet_id,mode=sntwitter.TwitterTweetScraperMode.SINGLE).get_items():
        tweet = str(item.content)
        break

    # some tweets have a https link at the end (when replying to a tweet), remove all https links
    tweet = remove_url(tweet)
    
    return tweet

def query_user(user):    
    query = f'(from:{user} exclude:replies)'    # query the user
    tweets = []
    limit = 30  # tweets limit
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == limit:
            break
        else:
            tweets.append(str([tweet.content])) # only append the content of the tweet
    
    # The API appends the tweets inside brakets [], remove those brakets
    for i in range(len(tweets)):
        tweets[i] = remove_url(tweets[i])
        tweets[i] = tweets[i].replace("[","")
        tweets[i] = tweets[i].replace("]","")

    return tweets



