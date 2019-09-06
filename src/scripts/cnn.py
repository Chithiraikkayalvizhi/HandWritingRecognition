import tensorflow as tf
import mongo
import numpy as np
import keras
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from pickle import load
from pickle import dump

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

(x_symbol_train, y_symbol_train) = mongo.get_train_data_from_db()
(x_symbol_test, y_symbol_test) = mongo.get_test_data_from_db()
(x_symbol_validation, y_symbol_validation) = mongo.get_validation_data_from_db()

## MNIST

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

## CROHME

x_symbol_train = x_symbol_train.reshape(x_symbol_train.shape[0], 50, 50, 1)
x_symbol_test = x_symbol_test.reshape(x_symbol_test.shape[0], 50, 50, 1)
x_symbol_validation = x_symbol_validation.reshape(x_symbol_validation.shape[0], 50, 50, 1)

symbol_input_shape = (50, 50, 1)

unique_values = np.unique(y_symbol_train)
print(len(unique_values))

x_symbol_train = x_symbol_train.astype('float32')
x_symbol_test = x_symbol_test.astype('float32')
x_symbol_validation = x_symbol_validation.astype('float32')

label_encoder = LabelEncoder()
y_symbol_train_int_encoded = label_encoder.fit_transform(y_symbol_train)
y_symbol_test_int_encoded = label_encoder.transform(y_symbol_test)
y_symbol_validation_int_encoded = label_encoder.transform(y_symbol_validation)

y_symbol_train_int_encoded = y_symbol_train_int_encoded.reshape(len(y_symbol_train_int_encoded), 1)
y_symbol_test_int_encoded = y_symbol_test_int_encoded.reshape(len(y_symbol_test_int_encoded), 1)
y_symbol_validation_int_encoded = y_symbol_validation_int_encoded.reshape(len(y_symbol_validation_int_encoded), 1)

onehot_encoder = OneHotEncoder(sparse=False)

y_symbol_train_onehot_encoded = onehot_encoder.fit_transform(y_symbol_train_int_encoded)
y_symbol_test_onehot_encoded = onehot_encoder.transform(y_symbol_test_int_encoded)
y_symbol_validation_onehot_encoded = onehot_encoder.transform(y_symbol_validation_int_encoded)


x_symbol_train /= 255
x_symbol_test /= 255
x_symbol_validation /= 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

print('x_symbol_train shape:', x_symbol_train.shape)
print('Number of images in x_symbol_train', x_symbol_train.shape[0])
print('Number of images in x_symbol_test', x_symbol_test.shape[0])
print('Number of images in x_symbol_validation', x_symbol_validation.shape[0])

model = Sequential()

model.add(Conv2D(50, kernel_size=(3,3), input_shape=symbol_input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(101,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(x=x_symbol_train,y=y_symbol_train_int_encoded, epochs=20, batch_size=15, validation_data=(x_symbol_validation, y_symbol_validation_int_encoded))

scores = model.evaluate(x_symbol_test, y_symbol_test_int_encoded)

print(scores)

model.save("epoch10_1.h5")
with open('epoch10_1.hisory', 'wb') as handle:
    dump(history.history, handle)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("done")

