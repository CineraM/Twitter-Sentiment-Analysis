from flask import Flask, render_template, url_for, request, redirect, session, flash
import qtweet as qt

app = Flask(__name__)
app.secret_key = "admin"

@app.route('/', methods=['POST', 'GET'])
@app.route('/home')



@app.route('/home', methods=['POST', 'GET'])
def home_page():
    if request.method == 'POST':
        try:    # succesful query 
            input_tweet = qt.query_tweet(str(request.form['tweet']))  
            # NEURAL NETWORK LOGIC SHOULD GO HERE
            
            # PLACE HOLDER LOGIC
            session["tweet"] = input_tweet
            session["sentiment_color"] = "green"
            session["sentiment"] = "POSITIVE"
            # PLACE HOLDER LOGIC
            
            return redirect(url_for('result'))
        except: # query failed 
            flash('Invalid tweet', 'danger')
            return render_template('home.html')

    return render_template('home.html')

@app.route('/result', methods=['POST','GET'])
def result():
    
    if request.method == 'POST':
        try:    # succesful query 
            input_tweet = qt.query_tweet(str(request.form['tweet']))  
            # NEURAL NETWORK LOGIC SHOULD GO HERE
            
            # PLACE HOLDER LOGIC
            session["tweet"] = input_tweet
            session["sentiment_color"] = "green"
            session["sentiment"] = "POSITIVE"
            # PLACE HOLDER LOGIC
            
            return redirect(url_for('result'))
        except: # query failed 
            flash('Invalid tweet', 'danger')
            return render_template('result.html')
        
    return render_template('result.html')


@app.route('/index', methods=['POST','GET'])
def index():
    return render_template('index.html')     

if __name__ == '__main__':
    app.run(debug=True)