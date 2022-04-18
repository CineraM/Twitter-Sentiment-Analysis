url = "https://twitter.com/elonmusk/status/1514681422212128770?cxt=HHwWhICz7c6unYUqAAAA"

last_forwardslash = url.rindex('/')
tweet_id = url[last_forwardslash+1:]

lastdigit = 0
for i in range(len(tweet_id)):
    if not tweet_id[i].isnumeric():
        lastdigit = i
        break

tweet_id = tweet_id[:lastdigit]