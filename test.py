import tensorflow as tf
import getTweets as gt


def load_model(filename):
    return tf.keras.models.load_model(filename)

#load in the trained model
model = load_model("model.tf")


def predict_tweet(tweet, model):
    
    string = str.encode(tweet)                  # format the tweet for the network
    encoded_tweet = [string]
    yfit = model(tf.constant(encoded_tweet))
    print(yfit)
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


train_testing = tf.data.TextLineDataset("emotions_test.csv").skip(1)
x_train = []
sent_list = []
result_sent = []
i = 0
for line in train_testing:
    split_line = tf.strings.split(line, ",", maxsplit=2)
    x_train.append(split_line[2].numpy())
    sent_list.append(split_line[1].numpy().decode('UTF-8'))
    x = x_train[i].decode('UTF-8')
    result_sent.append(predict_tweet(x, model))
    i += 1
    if(i >= 79):
        break

#loop variable to loop through the x_train and count to get accuracy
k = 0
amt_correct = 0
#calculate accuracy for the specified emotion
for sent in sent_list:
    print(sent)
    print ("Result " + result_sent[k])
    if (sent == result_sent[k]):
        amt_correct += 1
    k += 1

print(amt_correct)
acc = (amt_correct/len(sent_list))*100
print(acc)
print("Accuracy: "+ str(acc))


    
# a = "Just got back from seeing @GaryDelaney in Burslem. AMAZING!! Face still hurts from laughing so much #hilarious"

# print(predict_tweet(a, model))