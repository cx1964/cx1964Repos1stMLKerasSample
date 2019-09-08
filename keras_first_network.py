# Filenaam: keras_first_network.py 
# Functie: 1e Machine Learning python3 script obv python library keras 
# Zie artikel: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# Input file: pima-indians-diabetes.csv
# 
# Opmerking: 
# Input file moet in dezelfde directory staan als keras_first_network.py
# You can then run the Python file as a script from your command line (command prompt)
# as follows:
#             python keras_first_network.py
#
# Running this example, you should see a message for each of the 150 epochs printing the loss and accuracy,
# followed by the final evaluation of the trained model on the training dataset.
#
# It takes about 10 seconds to execute on my workstation running on the CPU.
#
# Ideally, we would like the loss to go to zero and accuracy to go to 1.0 (e.g. 100%). This is not possible
# for any but the most trivial machine learning problems. Instead, we will always have some error in our model.
# The goal is to choose a model configuration and training configuration that achieve the lowest loss and
# highest accuracy possible for a given dataset.
#
# Voorbeeld output:
# 768/768 [==============================] - 0s 63us/step - loss: 0.4817 - acc: 0.7708
# Epoch 147/150
# 768/768 [==============================] - 0s 63us/step - loss: 0.4764 - acc: 0.7747
# Epoch 148/150
# 768/768 [==============================] - 0s 63us/step - loss: 0.4737 - acc: 0.7682
# Epoch 149/150
# 768/768 [==============================] - 0s 64us/step - loss: 0.4730 - acc: 0.7747
# Epoch 150/150
# 768/768 [==============================] - 0s 63us/step - loss: 0.4754 - acc: 0.7799
# 768/768 [==============================] - 0s 38us/step
# Accuracy: 76.56




### 0. De libraries ###
#######################
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# A Keras neural network (deep learning) programma bestaat uit 6 stappen:
#    1. load data.
#    2. define a neural network in Keras.
#    3. compile a Keras model using the efficient numerical backend.
#    4. train a model on data.
#    5. evaluate a model on data.
#    6. predictions with the model.

# The data will be stored in a 2D array where the first dimension is rows
# and the second dimension is columns, e.g. [rows, columns].

### 1. Load Data ###
####################
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')


# Once the CSV file is loaded into memory,
# we can split the columns of data into input and output variables.

# split into input (X) and output (y) variables

# We can split the array into two arrays by selecting subsets of columns using
# the standard NumPy slice operator or “:” We can select the first 8 columns from index 0 to index 7
# via the slice 0:8. We can then select the output column (the 9th variable) via index 8.
# Zie ook:
# https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
X = dataset[:,0:8]
y = dataset[:,8]

### 2. Define Keras Model ###
#############################
# Models in Keras are defined as a sequence of layers.
# We create a Sequential model and add layers one at a time until we are happy with
# our network architecture.

# The first thing to get right is to ensure the input layer has the right number of
# input features. This can be specified when creating the first layer with the
# input_dim argument and setting it to 8 for the 8 input variables.

# How do we know the number of layers and their types?
# This is a very hard question. There are heuristics that we can use and often
# the best network structure is found through a process of trial and error experimentation
# (I explain more about this here
# https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/
# ).
# Generally, you need a network large enough to capture the structure of the problem.
# In this example, we will use a fully-connected network structure with three layers.
#
# We are now ready to define our neural network model.
#
# Fully connected layers are defined using the Dense class.
# We can specify the number of neurons or nodes in the layer as the first argument,
# and specify the activation function using the activation argument.
#
# We will use the rectified linear unit activation function referred to as ReLU on
# the first two layers and the Sigmoid function in the output layer.
#
# It used to be the case that Sigmoid and Tanh activation functions were preferred for all layers.
# These days, better performance is achieved using the ReLU activation function. We use a sigmoid
# on the output layer to ensure our network output is between 0 and 1 and easy to map to either
# a probability of class 1 or snap to a hard classification of either class with a default threshold
# of 0.5.
#
# We can piece it all together by adding each layer:
#
#    The model expects rows of data with 8 variables (the input_dim=8 argument)
#    The first hidden layer has 12 nodes and uses the relu activation function.
#    The second hidden layer has 8 nodes and uses the relu activation function.
#    The output layer has one node and uses the sigmoid activation function.

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Note:
# the most confusing thing here is that the shape of the input to the model
# is defined as an argument on the first hidden layer.
# This means that the line of code that adds the first Dense layer is doing 2 things:
#
#   1. defining the input or visible layer
#   2. and the first hidden layer.

### 3. Compile Keras Model ###
##############################
# Now that the model is defined, we can compile it.
#
# Compiling the model uses the efficient numerical libraries under the covers
# (the so-called backend) such as Theano or TensorFlow. The backend automatically chooses
# the best way to represent the network for training and making predictions to run on your
# hardware, such as CPU or GPU or even distributed.
#
# When compiling, we must specify some additional properties required when training the network.
# Remember training a network means finding the best set of weights to map inputs to outputs in
# our dataset.
#
# We must specify the loss function to use to evaluate a set of weights, the optimizer is used to
# search through different weights for the network and any optional metrics we would like to collect
# and report during training.
#
# In this case, we will use cross entropy as the loss argument. This loss is for a binary
# classification problems and is defined in Keras as “binary_crossentropy“. You can learn more
# about choosing loss functions based on your problem here (
# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
# ).
# We will define the optimizer as the efficient stochastic gradient descent algorithm “adam“.
# This is a popular version of gradient descent because it automatically tunes itself and gives
# good results in a wide range of problems. To learn more about the Adam version of
# stochastic gradient descent see(
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# ) 
#
# Finally, because it is a classification problem, we will collect and report the classification
# accuracy, defined via the metrics argument.
#
# Compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

### 4. Fit Keras Model ###
##########################
# We have defined our model and compiled it ready for efficient computation.
#
# Now it is time to execute the model on some data.
#
# We can train or fit our model on our loaded data by calling the fit() function on the model.
#
# Training occurs over epochs and each epoch is split into batches.
#
#    Epoch: One pass through all of the rows in the training dataset.
#    Batch: One or more samples considered by the model within an epoch before weights are updated.
#
# One epoch is comprised of one or more batches, based on the chosen batch size and the model is
# fit for many epochs. For more on the difference between epochs and batches, see the post(
# https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
# )
#
# The training process will run for a fixed number of iterations through the dataset called epochs,
# that we must specify using the epochs argument. We must also set the number of dataset rows that
# are considered before the model weights are updated within each epoch, called the batch size and
# set using the batch_size argument.
#
# For this problem, we will run for a small number of epochs (150) and use a relatively small batch
# size of 10. This means that each epoch will involve (150/10) 15 updates to the model weights.
#
# These configurations can be chosen experimentally by trial and error. We want to train the model
# enough so that it learns a good (or good enough) mapping of rows of input data to the output
# classification. The model will always have some error, but the amount of error will level out
# after some point for a given model configuration. This is called model convergence.

# Fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
#
# This is where the work happens on your CPU or GPU.
# No GPU is required for this example, but if you’re interested in how to run large models on
# GPU hardware cheaply in the cloud, see this post (
# https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/    
# )

### 5. Evaluate Keras Model ###
###############################
#
# We have trained our neural network on the entire dataset and we can evaluate the performance of the network
# on the same dataset.
#
# This will only give us an idea of how well we have modeled the dataset (e.g. train accuracy), but no idea of
# how well the algorithm might perform on new data. We have done this for simplicity, but ideally, you could
# separate your data into train and test datasets for training and evaluation of your model.
#
# You can evaluate your model on your training dataset using the evaluate() function on your model and pass it
# the same input and output used to train the model.
#
# This will generate a prediction for each input and output pair and collect scores, including the average loss
# and any metrics you have configured, such as accuracy.
#
# The evaluate() function will return a list with two values. The first will be the loss of the model on the
# dataset and the second will be the accuracy of the model on the dataset. We are only interested in reporting
# the accuracy, so we will ignore the loss value.
#
# Evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

### Tie It All Together ###
###########################
# Zie hierboven al gedaan.

### Extra opmerkingen mbt het script ###
########################################
#
# Note, if you try running this example in an IPython or Jupyter notebook you may get an error.
#
# The reason is the output progress bars during training. You can easily turn these off by setting verbose=0
# in the call to the fit() and evaluate() functions,
# for example:
#
#               # fit the keras model on the dataset without progress bars
#               model.fit(X, y, epochs=150, batch_size=10, verbose=0)
#               # evaluate the keras model
#               _, accuracy = model.evaluate(X, y, verbose=0)
#
#Note, the accuracy of your model will vary.
#
#Neural networks are a stochastic algorithm, meaning that the same algorithm on the same data can train a different model with different skill each time the code is run. This is a feature, not a bug. You can learn more about this in the post:
#
#    Embrace Randomness in Machine Learning
#
#The variance in the performance of the model means that to get a reasonable approximation of how well your model is performing, you may need to fit it many times and calculate the average of the accuracy scores. For more on this approach to evaluating neural networks, see the post:
#
#    How to Evaluate the Skill of Deep Learning Models
#
# For example, below are the accuracy scores from re-running the example 5 times:
#
#              Accuracy: 75.00
#              Accuracy: 77.73
#              Accuracy: 77.60
#              Accuracy: 78.12
#              Accuracy: 76.17
#
# We can see that all accuracy scores are around 77% and the average is 76.924%

### 6. Make Predictions ###
###########################
#
# The number one question I get asked is:
#
#    "After I train my model, how can I use it to make predictions on new data?"
#
# Great question.
#
# We can adapt the above example and use it to generate predictions on the training dataset, pretending
# it is a new dataset we have not seen before.
#
# Making predictions is as easy as calling the predict() function on the model. We are using a sigmoid
# activation function on the output layer, so the predictions will be a probability in the range between
# 0 and 1. We can easily convert them into a crisp binary prediction for this classification task by rounding
# them.
#
# For example:
#
#              # make probability predictions with the model
#              predictions = model.predict(X)
#              # round predictions 
#              rounded = [round(x[0]) for x in predictions]

# Make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

# Running the example does not show the progress bar as before as we have set the verbose argument to 0.
#
# After the model is fit, predictions are made for all examples in the dataset, and the input rows and
# predicted class value for the first 5 examples is printed and compared to the expected class value.
#
# We can see that most rows are correctly predicted. In fact, we would expect about 76.9% of the rows to
# be correctly predicted based on our estimated performance of the model in the previous section.
#
# [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0] => 0 (expected 1)
# [1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0] => 0 (expected 0)
# [8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0] => 1 (expected 1)
# [1.0, 89.0, 66.0, 23.0, 94.0, 28.1, 0.167, 21.0] => 0 (expected 0)
# [0.0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33.0] => 1 (expected 1) 
# 
# If you would like to know more about how to make predictions with Keras models, see the post(
# https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/   
# )   