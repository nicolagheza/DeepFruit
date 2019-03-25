from network import cnn
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.utils import np_utils
import tensorflow as tf
import pickle

tf.app.flags.DEFINE_string('model_path', 'model.h5', 'Path to model file')
tf.app.flags.DEFINE_string('data_path', 'C:\Code\DeepFruit\Fruit_Database', 'Path to data folder')
FLAGS = tf.app.flags.FLAGS

no_of_classes = 15

seed = 42

pickle_in = open("{}\X.pickle".format(FLAGS["data_path"].value),"rb")
X = pickle.load(pickle_in)

pickle_in = open("{}\y.pickle".format(FLAGS["data_path"].value),"rb")
y = pickle.load(pickle_in)
y = np_utils.to_categorical(y, no_of_classes) 

input_shape = (100, 100, 1)

model = load_model(FLAGS["model_path"].value)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

loss, acc = model.evaluate(X_train, y_train)
val_loss, val_acc = model.evaluate(X_test, y_test)

print("Accuracy: {}, Loss: {}".format(acc, loss))
print("Validation Accuracy: {}, Validation Loss: {}".format(val_acc, val_loss))