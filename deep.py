import tensorflow_datasets
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

dataset, info = tensorflow_datasets.load(
    'imdb_reviews', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

x_train = []
y_train = []

for sample, label in train_dataset.take(5000):
    x_train.append(sample.numpy())
    y_train.append(label.numpy())

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print(x_train[0])
print(y_train[0])

x_test = []
y_test = []

for sample, label in test_dataset.take(5000):
    x_test.append(sample.numpy())
    y_test.append(label.numpy())

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
vocab[:100]

print(x_train[:2])
vectorized = vectorize_layer(x_train[:2]).numpy()
print(vectorized)

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

history = model.fit(x=x_train, y=y_train, epochs=10,
                    batch_size=64, validation_data=(x_test, y_test))
