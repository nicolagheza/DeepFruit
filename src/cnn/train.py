from network import cnn
from keras.utils import np_utils
from tensorflow.keras.callbacks import TensorBoard

import pickle

pickle_in = open("C:/Users/nicol/Code/DeepFruit/Fruit_Database/X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("C:/Users/nicol/Code/DeepFruit/Fruit_Database/y.pickle","rb")
y = pickle.load(pickle_in)
y = np_utils.to_categorical(y, no_of_classes) 

no_of_classes = 15

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
            NAME = "Fruit_Database-CNN-{}x{}x{}x{}-{}x{}".format(
            cnn_filter[0], cnn_filter[1], cnn_filter[2], cnn_filter[3],
            fc_units[0][0], fc_units[0][1])
            print("Current CNN config: {}".format(NAME))
            print_config(cnn_filter, fc_units[0])
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            model = cnn(input_shape, cnn_filters[0], fc_units[0], no_of_classes)
        elif i == 8:
            NAME = "Fruit_Database-CNN-{}x{}x{}x{}-{}x{}".format(
            cnn_filter[0], cnn_filter[1], cnn_filter[2], cnn_filter[3],
            fc_units[1][0], fc_units[1][1])
            print("Current CNN config: {}".format(NAME))
            print_config(cnn_filter, fc_units[1])
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            model = cnn(input_shape, cnn_filters[0], fc_units[0], no_of_classes)
        else:
            NAME = "Fruit_Database-CNN-{}x{}x{}x{}-{}x{}".format(
            cnn_filter[0], cnn_filter[1], cnn_filter[2], cnn_filter[3],
            fc_units[2][0], fc_units[2][1])
            print("Current CNN config: {}".format(NAME))
            print_config(cnn_filter, fc_units[2])
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            model = cnn(input_shape, cnn_filters[0], fc_units[0], no_of_classes)
            
        history = model.fit(X, y, batch_size=32, epochs=1, validation_split=0.3, callbacks=[tensorboard])
        model.save_weights(NAME + ".h5")


if if __name__ == "__main__":
    train()