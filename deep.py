from posixpath import split
import tensorflow_datasets
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


def getDataset(datasetFile):
    for line in datasetFile.skip(1):
        split_line = tf.strings.split(line, ",", maxsplit=2)
        review = split_line[2]

        print(review)
        print(split_line)
        # tokenized_text = tokenizer.tokenize(review.numpy().lower())

        # for word in tokenized_text:
        #     if word not in frequencies:
        #         frequencies[word] = 1
        #     else:
        #         frequencies[word] += 1

        #     # if we've reached the threshold
        #     if frequencies[word] == threshold:
        #         vocabulary.update(tokenized_text)


def getLabel(label):
    if label == str.encode("joy"):
        return 0
    elif label == str.encode("sadness"):
        return 1
    elif label == str.encode("fear"):
        return 2
    elif label == str.encode("anger"):
        return 3


"""
Load Data
"""
train_dataset = tf.data.TextLineDataset("emotions_train.csv").skip(1)
test_dataset = tf.data.TextLineDataset("emotions_test.csv").skip(1)

"""
Put data in arrays
"""

x_train = []
y_train = []

for line in train_dataset:
    split_line = tf.strings.split(line, ",", maxsplit=2)
    x_train.append(split_line[2].numpy())
    y_train.append(getLabel(split_line[1].numpy()))

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print(x_train[0])
print(y_train[0])

x_test = []
y_test = []

for line in test_dataset:
    split_line = tf.strings.split(line, ",", maxsplit=2)
    x_test.append(split_line[2].numpy())
    y_test.append(getLabel(split_line[1].numpy()))

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print(x_test[0])
print(y_test[0])

max_words = 1000
#max_len = 100
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_words)  # , output_sequence_length=max_len)

vectorize_layer.adapt(x_train)

vocab = np.array(vectorize_layer.get_vocabulary())
# print(vocab[:100])

# print(x_train[:2])
vectorized = vectorize_layer(x_train[:2]).numpy()
# print(vectorized)

model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(
        input_dim=max_words, output_dim=64, mask_zero=True),
    tf.keras.layers.SimpleRNN(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

history = model.fit(x=x_train, y=y_train, epochs=100,
                    batch_size=64, validation_data=(x_test, y_test))

# model.save("my_model.tf")

# model = load_model("my_model.tf")

string = str.encode(
    "I swear to god if I dont win next game i will blow this place up")
test = [string]
# vectorize_layer.adapt(test)

yfit = model(tf.constant(test))
yfit = yfit.numpy().argmax(axis=1)

print(yfit)
