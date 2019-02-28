from network import cnn
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
 
import pickle

no_of_classes = 15

pickle_in = open("C:/Users/nicol/Code/DeepFruit/Fruit_Database/X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("C:/Users/nicol/Code/DeepFruit/Fruit_Database/y.pickle","rb")
y = pickle.load(pickle_in)
y = np_utils.to_categorical(y, no_of_classes) 

input_shape = (100, 100, 1)

model = cnn(input_shape, no_of_classes)
model.load_weights('my_model_2.h5')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

scores = model.evaluate(X_test, y_test, verbose=1)

print("Scores: ", scores)