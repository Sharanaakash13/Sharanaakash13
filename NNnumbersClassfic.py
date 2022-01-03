import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

'''Loading the dataset'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, x_test.shape)
# print(y_train[10])
# plt.matshow(x_train[10])
# plt.show()

'''Normalize'''
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


'''Converting 2D array to 1D array'''
X_train_flattened = x_train.reshape(len(x_train), 28*28)
X_test_flattened = x_test.reshape(len(x_test), 28*28)
print(X_train_flattened)

# other way of flattening  and creating a NN using inbuilt function
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())    # flattens the image    # input layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))    # hidden layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))    # hidden layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer


# ''' Creating the neural network layers'''
# model = keras.Sequential([
#     tf.keras.layers.Dense(50, activation='sigmoid'),
#     tf.keras.layers.Dense(50, activation='sigmoid')
# ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1)

'''Evaluation'''
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

'''Saving model'''
model.save('handwritten')
new_model = tf.keras.models.load_model('handwritten')

'''Prediction'''
prediction = new_model.predict([x_test])
print(prediction)

predict_array = [np.argmax(i) for i in prediction]


'''Confusion matrix'''
cm = tf.math.confusion_matrix(labels=y_test, predictions=predict_array)
print(cm)
# heatmap
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
