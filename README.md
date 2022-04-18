# Intro to A.I Final Project
Members: Nathaniel Aldino, Daniel Arvelo, Raymond Gillies, Matias Cinera, Devin Parmet

## How to run
### Application Dependencies
     pip install snscrape
     pip install numpy
     pip install Flask
     pip install tensorflow

#### Application Structure
```
├── ...
├── model.tf             # Trained network saved on disk
├── Templates            # Trained network saved on disk
│   ├── base.html        # base html          
│   ├── t_home.html      # tweets home 
│   ├── t_result.html    # tweets results  
│   ├── u_home.html      # user tweets home  
│   └── u_home.html      # user tweets results 
├── app.py               # flask application & load NN
├── deep.py              # Train and save NN
├── getTweets.py         # Trained network saved on disk
├── emotions_train.csv   # Trainning data for NN
├── emotions_test.csv    # Testing data for NN
└── 
```

#### Optional: Train the Neural Network by compiling "deep.py"
#### Compile the Flask application "app.py"
     python deep.py
     python app.py

In our project, we aimed to evaluate text data by generating a model that can determine the emotion portrayed by the text in question. This is a popular machine lea!
rning technique and is more commonly known as sentiment analysis.  


[app1](https://user-images.githubusercontent.com/64340009/163760724-8e8ee5a7-bb74-43f6-89e7-94ce02ab74af.png)  


![app2](https://user-images.githubusercontent.com/64340009/163760746-b7bbab87-ca08-44ce-94d2-b4640a0901b8.png)  



