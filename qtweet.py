# https://www.youtube.com/watch?v=jtIMnmbnOFo <-- thank go
import snscrape.modules.twitter as sntwitter

def query_tweet(url):
    last_forwardslash = url.rindex('/')

    tweet_id = url[last_forwardslash+1:]

    #content = []
    tweet = ""
    # loop that gets content from a tweet --> user, tweet, followers...
    for item in sntwitter.TwitterTweetScraper(tweetId=tweet_id,mode=sntwitter.TwitterTweetScraperMode.SINGLE).get_items():
        #content.append([item.user.username, item.content])
        print(item)
        tweet = str(item.content)
        break

    #print(tweet)
    # some tweets have a https link (when replying to a tweet), remove all https links
    try:
        while True:
            html_index = tweet.rindex("https")
            tweet = tweet[:html_index]
    except:
        None
        # if the tweet doesnt have a https str, rindex will 
        # crash the program. There is probably a better solution for this 
    return tweet

#print(query_tweet("https://twitter.com/WHO/status/1511827080043978756"))