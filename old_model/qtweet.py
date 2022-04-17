# https://www.youtube.com/watch?v=jtIMnmbnOFo <-- thank go
import snscrape.modules.twitter as sntwitter

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
    try:
        while True:
            html_index = tweet.rindex("https")
            tweet = tweet[:html_index]
    except:
        None
        # if the tweet doesnt have a https str, rindex will 
        # crash the program. There is probably a better solution for this 
        # REWORK THIS LOGIC
    return tweet

# testing
#print(query_tweet("https://twitter.com/WHO/status/1511827080043978756"))
#print(query_tweet("https://twitter.com/elonmusk/status/1514681422212128770?cxt=HHwWhICz7c6unYUqAAAA"))