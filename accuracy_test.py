import tensorflow as tf
import getTweets as gt


def load_model(filename):
    return tf.keras.models.load_model(filename)

#load in the trained model
model = load_model("model.tf")

# this is the prediction function that uses the already model and determines the sentiment
def predict_tweet(tweet, model):
    
    string = str.encode(tweet)                  # format the tweet for the network
    encoded_tweet = [string]
    yfit = model(tf.constant(encoded_tweet))
    # print(yfit)
    yfit = yfit.numpy().argmax(axis=1)          # fit the tweet on the network

    sentiment = ""
    if yfit == 0:
        sentiment = "joy"
    elif yfit == 1:
        sentiment = "sadness"
    elif yfit == 2:
        sentiment = "fear"
    elif yfit == 3:
        sentiment = "anger"
    
    return sentiment

# This is the smaller testing set used to evaluate accuracy
testing = tf.data.TextLineDataset("emotions_test_small.csv").skip(1)
# list to hold the tweets
x_train = []
# list to hold the sentiments of the tweets in test
sent_list = []
# list to hold the predited sentiment by the model
result_sent = []

i = 0
for line in testing:
    # split the csv
    split_line = tf.strings.split(line, ",", maxsplit=2)
    # specify emotion here to get that accuracy
    # if the line in the dataset is the specified emotion, send the tweet to the model 
    # and add the actual sentiment and predicted sentiment to respective lists
    if(split_line[1].numpy().decode('UTF-8') == 'anger'):
        x_train.append(split_line[2].numpy())
        sent_list.append(split_line[1].numpy().decode('UTF-8'))
        x = x_train[i].decode('UTF-8')
        i += 1
        result_sent.append(predict_tweet(x, model))
   

#loop variable to loop through the x_train and count to get accuracy
k = 0
predicted_correct = 0
#calculate accuracy for the specified emotion
for sent in sent_list:
    print(sent)
    print ("Result " + result_sent[k])
    if (sent == result_sent[k]):
        predicted_correct += 1
    k += 1

#calculate the accuracy
acc = (predicted_correct/len(sent_list))*100
print(acc)
print("Accuracy: "+ str(acc))
