# The code was run on google colab so it needs access to google
# drive for data interface. The following code is to access the
# drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
​
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
from google.colab import drive
drive.mount('/content/gdrive')

# Import the necessary libraries for model training and testing
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score

# Define a simple model with one hidden layer with relu activation
# and output with softmax activation
def SimpleModel():
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=[28, 28]),
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(25, activation=tf.nn.softmax)
	])
	return model

# Visualize the loss as a function of iteration (Derived from Keras tutorial)
def visualize_loss(h):
    loss = h.history["loss"]
    plt.figure(figsize=(10, 10))
    plt.plot(loss)
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.title("Plot of loss against number of iterations")
    plt.grid()


# Define the location of train and test data and import them
train_path = "dataset/sign_mnist_train.csv"
test_path = "dataset/sign_mnist_test.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Split the entire data into train and test data variables
train_x, train_y = train_data.drop(['label'], axis=1).values, train_data['label'].values
test_x, test_y = train_data.drop(['label'], axis=1).values, train_data['label'].values

# Pre process the data and normalize the data
train_x = train_x.reshape(-1, 28, 28) / 255.0
test_x = test_x.reshape(-1, 28, 28) / 255.0

# Convert the y data to 25 categorical data using keras's to_categorical
train_y = keras.utils.to_categorical(train_y, 25)

# Instantiate the model
model = SimpleModel()
print(model.summary())

# Compile the model with Adam optimizer and optimize on loss
model.compile(optimizer="Adam", metrics = ["accuracy"], loss="categorical_crossentropy")

# Training the model with 10 epochs
EPOCHS = 10
history = model.fit(train_x, train_y, epochs=EPOCHS)
visualize_loss(history)

# predict the labels on the test data
predict = model.predict(test_x)
pred_labels = np.argmax(predict, axis=1)

# Get the mean per class accuracy
conf = confusion_matrix(pred_labels, test_y)
conf = np.divide(conf, conf.max())
mean_per_class = np.mean(np.diagonal(conf))
print(mean_per_class)

# Calculate the recall and precision
recall = recall_score(pred_labels, test_y, average='macro') 
precision = precision_score(pred_labels, test_y, average='macro') 
print(recall, precision)

# Display the label predicted for randomly selected frame
random_index = np.random.randint(0, len(test_x))
print("Index :", random_index)
print("Actual label:", labels[test_y[random_index]])
print("Predicted label:", labels[np.argmax(predict[random_index])])
print("Accuracy:", predict[random_index][np.argmax(predict[random_index])])
for i in np.arange(25):
    print("The probability that the image is {} = {:.3f}".format(labels[i], predict[random_index][i]))
plt.imshow(test_x[random_index], cmap="gray")

# Save the weights and model for offline use
model.save('gdrive/My Drive/Colab Notebooks/simple_nn.h5')
model.save_weights('gdrive/My Drive/Colab Notebooks/simple_nn_weights.h5')