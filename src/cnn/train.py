from network import cnn
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import pickle

seed = 42

tf.app.flags.DEFINE_string('data_path', 'C:\Code\DeepFruit\Fruit_Database', 'Path to data folder')
FLAGS = tf.app.flags.FLAGS
print (FLAGS.data_path)

pickle_in = open("{}\X.pickle".format(FLAGS.data_path),"rb")
X = pickle.load(pickle_in)

pickle_in = open("{}\y.pickle".format(FLAGS.data_path),"rb")
y = pickle.load(pickle_in)

no_of_classes = 15

y = np_utils.to_categorical(y, no_of_classes) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

input_shape = (100, 100, 1)

cnn_filters = [
    [16, 32, 64, 128], [8, 32, 64, 128], [32, 32, 64, 128], 
    [16, 16, 64, 128], [16, 64, 64, 128], [16, 32, 32, 128], 
    [16, 32, 128, 128], [16, 32, 64, 64], [16, 32, 64, 128], [16, 32, 64, 128]
]
fc_units = [[1024, 256], [512, 256], [1024, 512]]

def print_config(cnn_filters, fc_units):
    for cnn_filter in cnn_filters:
        print("Convolutional   | 5x5 | {}".format(cnn_filter))
    for fc_unit in fc_units:
        print("Fully connected |  -  | {}".format(fc_unit))

def train():
    for i, cnn_filter in enumerate(cnn_filters):
        if i < 8:
            NAME = "Fruit_Database-150x150-CNN-{}x{}x{}x{}-{}x{}".format(
            cnn_filter[0], cnn_filter[1], cnn_filter[2], cnn_filter[3],
            fc_units[0][0], fc_units[0][1])
            print("Current CNN config: {}".format(NAME))
            print_config(cnn_filter, fc_units[0])
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            model = cnn(input_shape, cnn_filters[0], fc_units[0], no_of_classes)
        elif i == 8:
            NAME = "Fruit_Database-150x150-CNN-{}x{}x{}x{}-{}x{}".format(
            cnn_filter[0], cnn_filter[1], cnn_filter[2], cnn_filter[3],
            fc_units[1][0], fc_units[1][1])
            print("Current CNN config: {}".format(NAME))
            print_config(cnn_filter, fc_units[1])
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            model = cnn(input_shape, cnn_filters[0], fc_units[0], no_of_classes)
        else:
            NAME = "Fruit_Database-150x150-CNN-{}x{}x{}x{}-{}x{}".format(
            cnn_filter[0], cnn_filter[1], cnn_filter[2], cnn_filter[3],
            fc_units[2][0], fc_units[2][1])
            print("Current CNN config: {}".format(NAME))
            print_config(cnn_filter, fc_units[2])
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            model = cnn(input_shape, cnn_filters[0], fc_units[0], no_of_classes)
            
        history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), callbacks=[tensorboard])
        model.save_weights(NAME + ".h5")


if __name__ == "__main__":
    train()
