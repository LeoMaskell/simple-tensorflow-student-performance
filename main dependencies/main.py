# regression model

# import dataset
import pandas as pd

df = pd.read_csv('dataset.csv')

df = df.drop("student_id", axis=1)
print(df.head())

# shuffle and split dataset
from sklearn.model_selection import train_test_split
import numpy as np

# old buggier method of formatting dataset...
"""X = df.drop("exam_score", axis=1)
Y = df["exam_score"]

X = pd.get_dummies(X)
X = np.array(X, dtype=np.float32)

Y = pd.get_dummies(Y)
Y = np.array(Y, dtype=np.float32)

trainX, testX = train_test_split(X, test_size=0.2) # doing it with x and y this way means that y will always = 0. 0. 0. 0. etc.
trainY, testY = train_test_split(Y, test_size=0.2)
"""
df = pd.get_dummies(df)

train, test = train_test_split(df, test_size=0.2)

trainX = train.drop('exam_score', axis=1)
trainY = train['exam_score']

testX = test.drop('exam_score', axis=1)
testY = test['exam_score']

trainX = np.array(trainX, dtype=np.float32)
trainY = np.array(trainY, dtype=np.float32)

testX = np.array(testX, dtype=np.float32)
testY = np.array(testY, dtype=np.float32)



print(trainX)
print(trainY)

print(testX)
print(testY)



print ('dataset stuff is finished')




# the actual net
import tensorflow as tf
from tensorflow import keras

from keras import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.optimizers import Adam

# define the model
model = Sequential([
	Input(shape=(trainX.shape[1],), name="input_layer"),
	Dense(13),
	Dense(1024, activation="relu"),
    Dense(512, activation="relu"),
	Dense(512, activation="selu"),
	Dense(512, activation="selu"),
	Dense(1024, activation="relu"),
	Dense(1, activation=None, name="output_Layer")
])


optimizer = "Adam"
# compile the model
model.compile(
        optimizer = optimizer,
		loss = "mae",
		metrics = ["mse"])


print('\n\n model ready for training \n\n')

print(model)
print(model.summary())

print("\n\n trainig starting to happen. \n\n")

# training
print(trainX.shape)
print(trainY.shape)
print(model.output_shape)


history = model.fit(
    trainX,
    trainY,
    epochs=40,
    # show logs.
    verbose=1,
    # Calculate validation results on 80% of the training data.
    validation_split = 0.8
)



# visualise data from training
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# uncomment if using a gui - will show a graph of training.

import matplotlib.pyplot as plt
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)


# testing
# use model.evaluate(...) to test with the model
loss, accuracy = model.evaluate(testX, testY)
print(f"test loss: {loss:4f}")
print(f"test accuracy: {accuracy:4f}")

# idk how u actually use it but i will look it up...
# to use the model.predict(features[...]) to use it
print(f"predicted: {model.predict(testX[:1])}")
print(f"actual: {testY[:1]}")
#TODO: export the model after training and make a program to use it
model.save('./model/modelV1.keras')