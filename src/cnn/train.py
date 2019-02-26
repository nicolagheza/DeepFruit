from network import cnn
from keras.utils import np_utils
from tensorflow.keras.callbacks import TensorBoard

import pickle

# Load training data
NAME = "Fruit_Database-CNN-128"

no_of_classes = 15

pickle_in = open("C:/Users/nicol/Code/DeepFruit/Fruit_Database/X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("C:/Users/nicol/Code/DeepFruit/Fruit_Database/y.pickle","rb")
y = pickle.load(pickle_in)
y = np_utils.to_categorical(y, no_of_classes) 

input_shape = (100, 100, 1)

# Setup TensorBoard for Logs
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# Create Model and Start training
model = cnn(input_shape, no_of_classes)

model.summary()

history = model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3, callbacks=[tensorboard])