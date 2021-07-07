# The code was run on google colab so it needs access to google
# drive for data interface. The following code is to access the
# drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
from google.colab import drive
drive.mount('/content/gdrive')


# Import the libraries of keras for training and testin the model
import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import plot_model

# Other libraries
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# Make an Image generator for reading the data from the directory
# Reading with loop is expensive and takes a very long time 
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

# Define the number of samples
nb_train_samples = 87000

# Create a data generator that has read the data i.e. images as well as classes
# they belong to
train_generator = train_datagen.flow_from_directory("asldata/train", target_size = (200, 200), batch_size = 16, class_mode = "categorical")

# Create a VGG16 model pre-trained on imagenet and FC layer excluded
# The input size is (200, 200, 3)
model = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape = (200, 200, 3))

# Create model characteristic and save in file
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Make the first 5 convolution block of Vgg-16 nontrainable
for layer in model.layers[:5]:
    layer.trainable = False

# Create custom Fully connected layer at the end of the VGG convolution block
block_5output = model.output

# Input layer for the new FC layer
flattened = Flatten()(block_5output)

# A relu - dropout - relu sandwich layer as hidden layer
dense1 = Dense(1024, activation="relu")(flattened)
dropout_layer = Dropout(0.5)(dense1)
dense2 = Dense(1024, activation="relu")(dropout_layer)

# Output layer with 29 classes activated with softmax
predictions = Dense(29, activation="softmax")(dense2)


# CONCEPT DERIVED FROM https://github.com/anujshah1003/Transfer-Learning-in-keras---custom-data
# Instantiate the custome model defined above
custom_model = Model(input = model.input, output = predictions)

# Combile the model with Stochastic Gradient Descent Optimization 
custom_model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Create a model characteristic of the cursom model
plot_model(custom_model, to_file='model_custom_plot.png', show_shapes=True, show_layer_names=True)

# Train the network with the generator defined above and save the history
history = custom_model.fit_generator(train_generator, samples_per_epoch = 87000, epochs = 10)

# Save the trained complete model for offline use
custom_model.save("vgg16_trained.h5")

# Read the test data along with their classes using the Image data generator
test_generator = test_datagen.flow_from_directory("asldata/test", target_size = (200, 200), batch_size = 16, class_mode = "categorical")


# Get the prediction
pred = custom_model.predict_generator(test_generator, steps=1)
pred_label = pred.argmax(axis=0)