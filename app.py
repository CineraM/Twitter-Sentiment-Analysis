from flask import Flask, render_template, url_for, request, redirect, session, flash
import tensorflow as tf
import getTweets as gt

################################################
# lOADING NEURAL NETWORK & DEPENDENT FUNCTIONS #
################################################

def load_model(filename):
    return tf.keras.models.load_model(filename)

model = load_model("model.tf")

# Function that returns the sentiment given text
def predict_tweet(tweet, model):
    
    string = str.encode(tweet)                  # format the tweet for the network
    encoded_tweet = [string]
    yfit = model(tf.constant(encoded_tweet))
    yfit = yfit.numpy().argmax(axis=1)          # fit the tweet on the network

    sentiment = ""
    if yfit == 0:
        sentiment = "Joy"
    elif yfit == 1:
        sentiment = "Sadness"
    elif yfit == 2:
        sentiment = "Fear"
    elif yfit == 3:
        sentiment = "Anger"
    
    return sentiment

# Function that returns the sentiments of a list of tweets
def predict_user(tweets):
    sentiments = []
    for tweet in tweets:
        try:    # if the tweet only has a link it will creash the program. Just skip that tweet
            sentiments.append(predict_tweet(tweet, model))
        except:
            print("Failed to evaluate tweet")

    return sentiments

################################################
############# FLASK FUNCTIONS ##################
################################################
app = Flask(__name__)
app.secret_key = "admin"

@app.route('/', methods=['POST', 'GET'])
@app.route('/qhome')


@app.route('/qhome', methods=['POST', 'GET'])
def tweet_home():
    if request.method == 'POST':
        try:
            # User submitted a twitter link
            input_tweet = gt.query_tweet(str(request.form['tweet'])) # query the tweet
            
            sentiment = predict_tweet(input_tweet, model)   # get tweet sentiment
            session["tweet"] = input_tweet
            session["sentiment"] = sentiment

            if sentiment == "Joy":                          # assign different colors for sentiments
                session["sentiment_color"] = "green"
            elif sentiment == "Sadness":
                session["sentiment_color"] = "blue"
            elif sentiment == "Fear":
                session["sentiment_color"] = "grey"
            elif sentiment == "Anger":
                session["sentiment_color"] = "red"
         
            return redirect(url_for('tweet_result'))
        except: 
            # If the query failed display an error message 
            flash('Invalid tweet', 'danger')
            return render_template('t_home.html')

    return render_template('t_home.html')

@app.route('/tresult', methods=['POST','GET'])
def tweet_result():
    
    if request.method == 'POST':
        try:
            # same logic as tweet_home()
            input_tweet = gt.query_tweet(str(request.form['tweet']))  
            
            sentiment = predict_tweet(input_tweet, model)
            session["tweet"] = input_tweet
            session["sentiment"] = sentiment

            if sentiment == "Joy":
                session["sentiment_color"] = "green"
            elif sentiment == "Sadness":
                session["sentiment_color"] = "blue"
            elif sentiment == "Fear":
                session["sentiment_color"] = "grey"
            elif sentiment == "Anger":
                session["sentiment_color"] = "red"
         
            return redirect(url_for('tweet_result'))
        except: # query failed 
            flash('Invalid tweet', 'danger')
            return render_template('t_result.html')
        
    return render_template('t_result.html')

tweets_sentiments = []
@app.route('/uhome', methods=['POST', 'GET'])
def user_home():
    if request.method == 'POST':
        try:
            # User submitted a twitter user
            input_user = str(request.form['user'])  # get input from html
            user_tweets = gt.query_user(input_user) # get list of tweets from the user
            print(input_user)
            sentiments = predict_user(user_tweets)  # predict the sentiment from the tweets

            # get the most frequent sentiment
            max_sentiment = max(set(sentiments), key = sentiments.count)

            session["user"] = input_user
            session["sentiment"] = max_sentiment

            if max_sentiment == "Joy":              # assign different colors for sentiments
                session["sentiment_color"] = "green"
            elif max_sentiment == "Sadness":
                session["sentiment_color"] = "blue"
            elif max_sentiment == "Fear":
                session["sentiment_color"] = "grey"
            elif max_sentiment == "Anger":
                session["sentiment_color"] = "red"
 
            temp = []   # merge tweets and sentiments into a list 
            for i in range(len(sentiments)):
                temp.append([user_tweets[i], sentiments[i]])
            global tweets_sentiments # use a global variable to pass the list into the user_result html
            tweets_sentiments = temp

            return redirect(url_for('user_result'))
        except: # query failed 
            flash('Invalid twitter user', 'danger')
            return render_template('u_home.html')

    return render_template('u_home.html')

@app.route('/quser', methods=['POST','GET'])
def user_result():
    data = tweets_sentiments    # pass the tweets list into the user_results html
    return render_template('u_result.html', results=data)     

if __name__ == '__main__':
    app.run(debug=True)