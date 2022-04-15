from flask import Flask, render_template, url_for, request, redirect, session, flash
import qtweet as qt

app = Flask(__name__)
app.secret_key = "admin"

@app.route('/', methods=['POST', 'GET'])
@app.route('/qhome')


@app.route('/qhome', methods=['POST', 'GET'])
def tweet_home():
    if request.method == 'POST':
        try:    # succesful query 
            input_tweet = qt.query_tweet(str(request.form['tweet']))  
            # NEURAL NETWORK LOGIC SHOULD GO HERE
            
            # PLACE HOLDER LOGIC
            session["tweet"] = input_tweet
            session["sentiment_color"] = "green"
            session["sentiment"] = "POSITIVE"
            # PLACE HOLDER LOGIC
            
            return redirect(url_for('tweet_result'))
        except: # query failed 
            flash('Invalid tweet', 'danger')
            return render_template('t_home.html')

    return render_template('t_home.html')

@app.route('/tresult', methods=['POST','GET'])
def tweet_result():
    
    if request.method == 'POST':
        try:    # succesful query 
            input_tweet = qt.query_tweet(str(request.form['tweet']))  
            # NEURAL NETWORK LOGIC SHOULD GO HERE
            
            # PLACE HOLDER LOGIC
            session["tweet"] = input_tweet
            session["sentiment_color"] = "green"
            session["sentiment"] = "POSITIVE"
            # PLACE HOLDER LOGIC
            
            return redirect(url_for('tweet_result'))
        except: # query failed 
            flash('Invalid tweet', 'danger')
            return render_template('t_result.html')
        
    return render_template('t_result.html')


@app.route('/uhome', methods=['POST', 'GET'])
def user_home():
    if request.method == 'POST':
        try:    # succesful query 
            # NEURAL NETWORK LOGIC SHOULD GO HERE
            
            # PLACE HOLDER LOGIC
            input_user = str(request.form['user'])
            print(input_user)
            session["user"] = input_user
            session["sentiment_color"] = "yellow"
            session["sentiment"] = "Joyful"
            # PLACE HOLDER LOGIC
            
            return redirect(url_for('user_result'))
        except: # query failed 
            flash('Invalid twitter user', 'danger')
            return render_template('u_home.html')

    return render_template('u_home.html')

@app.route('/quser', methods=['POST','GET'])
def user_result():
    data = [["Just got back from seeing @GaryDelaney in Burslem. AMAZING!! Face still hurts from laughing so much #hilarious", "Joy"],
            ["Police Officers....should NOT have the right to just 'shoot' human beings without provocation. It's wrong.\n\n@ORConservative @MichaelaAngelaD", "Anger"],
            ["Currently unfollowing anything relating to disneyworld or Florida! #holidayblues #depressing #wantogoback ðŸ˜­ðŸ’”", "Sadness"],
            ["r U scared to present in front of the class? severe anxiety... whats That r u sad sometimes?? go get ur depression checked out IMEDIATELY!!!", "Fear"]]
    return render_template('u_result.html', results=data)     

if __name__ == '__main__':
    app.run(debug=True) 