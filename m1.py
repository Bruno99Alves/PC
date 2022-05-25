import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils.vis_utils import plot_model
import keras_tuner
from keras.preprocessing.image import load_img


def dataview(x_train,x_test):
  print('Train data shape:',x_train.shape)
  print('Test data shape:',x_test.shape)
  print('Number of training samples:',x_train.shape[0])
  print('Number of test samples:',x_test.shape[0])
  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap='gray')
  plt.show()

def graphs(t_acc, v_acc, t_loss,  v_loss):
  epochs = range(1, len(t_acc) + 1)
  plt.plot(epochs, t_acc, 'b', label='Training acc')
  plt.plot(epochs, v_acc, 'r', label='Validation acc')
  plt.title('Accuracy')
  plt.legend()
  plt.figure()

  plt.plot(epochs, t_loss, 'b', label='Training loss')
  plt.plot(epochs, v_loss, 'r', label='Validation loss')
  plt.title('Loss')
  plt.legend()
  plt.figure()

def mnist():
  #Loading Dataset
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  n_classes = 10
  print('N_CLASSES:',10)
  #Partitioning for Validation Dataset
  x_validation = x_train[-6000:]
  y_validation = y_train[-6000:]
  x_train = x_train[:-6000]
  y_train = y_train[:-6000]
  
  #Data Exploration
  dataview(x_train, x_test)

  #Reshape and Normalization 
  '''
  x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2],1).astype('float32')/255
  x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1]*x_validation.shape[2],1).astype('float32')/255
  x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2],1).astype('float32')/255
  '''
  #Normalization
  x_train = x_train.astype('float32')/255
  x_validation = x_validation.astype('float32')/255
  x_test = x_test.astype('float32')/255

  #Create Dataset Iterator
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
  validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation,y_validation))

  #Shuffling
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size) 
  validation_dataset = validation_dataset.batch(batch_size)
  
  return train_dataset, validation_dataset, x_test, y_test, n_classes

def build_CNN(hp):
    inputs = tf.keras.Input(shape=(28,28,1))
    model_optimizer = hp.Choice("model_optimizer", ["adam","rmsprop"])
    x = inputs
    for i in range(hp.Int("cnn_layers", 1, 3)):
            x = tf.keras.layers.Conv2D(
                hp.Int(f"filters_{i}", 32, 128, step=32),
                kernel_size=(3, 3),
                activation="relu",)(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)

    drop = 0.2
    if hp.Boolean("dropout"):
        x = tf.keras.layers.Dropout(drop)(x)

    outputs = tf.keras.layers.Dense(units=10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    if model_optimizer == "adam":
      adam = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate)
      model.compile(
        loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer = adam,
    )
    else:
      rmsprop = tf.keras.optimizers.RMSprop(learning_rate = hp_learning_rate)
      model.compile(
        loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer = rmsprop,
    )

    return model

model_name = 'mnist_cnn'
batch_size = 32
datashape = (28,28,1)
train_dataset, validation_dataset, x_test, y_test, n_classes = mnist()

hp = keras_tuner.HyperParameters()
hp.values["model_optimizer"] = "adam"

model = build_CNN(hp)
model.summary()

tuner = keras_tuner.Hyperband(
    build_CNN,
    overwrite=True,
    max_epochs= 10,
    objective="val_accuracy",
    directory="/tmp/tb",
    seed = 1234
)

tuner.search(
    train_dataset,
    validation_data = validation_dataset,
    callbacks=[tf.keras.callbacks.TensorBoard("/tmp/tb_logs")]
)

path= '/content/checkpoints'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
          filepath= path,
          save_weights_only=True,
          monitor='val_accuracy',
          mode='max',
          save_best_only=True)
best_hp = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hp)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
hist = model.fit(train_dataset, validation_data=validation_dataset, epochs=50,callbacks=[model_checkpoint_callback,stop_early])
t_acc = hist.history['accuracy']
v_acc = hist.history['val_accuracy']
t_loss = hist.history['loss']
v_loss = hist.history['val_loss']

with open('m1'"w") as o:
    o.write("t_acc:{}\n".format(t_acc))
    o.write("t_acc:{}\n".format(v_acc))
    o.write("t_acc:{}\n".format(t_loss))
    o.write("t_acc:{}\n".format(v_loss))