from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten

def cnn(input_shape, cnn_filters, fc_units, n_classes):
    model = Sequential()
    model.add(Conv2D(cnn_filters[0], kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv2D(cnn_filters[1], kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(cnn_filters[2], kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(cnn_filters[3], kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(fc_units[0], activation='relu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(fc_units[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

    return model    