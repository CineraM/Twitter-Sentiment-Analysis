import tensorflow as tf
import numpy as np


def getLabel(label):
    # numerical mapping of the labels

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

"""
Manipulate Data
"""
# Convert strings to arrays of numbers
max_words = 1000
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_words)

vectorize_layer.adapt(x_train)


"""
Deep learning
"""
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


"""
individual testing
"""
string = str.encode(
    "im smiling and singing all day")
test = [string]

yfit = model(tf.constant(test))
yfit = yfit.numpy().argmax(axis=1)

print(yfit)
