#####################################################################################
# Pre Existing Libraries
import os,sys
import random
import numpy                         as np
from   sklearn                       import preprocessing
import matplotlib.pyplot             as plt
from   sklearn.datasets              import make_classification
import tensorflow                    as tf
import time
import tflearn
#######################################################################################
# Some global parameters for the script
Faults = 4
dimension = 4
function_choose = "linear"
par = 0
learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 10
Faults = 4
#######################################################################################
# Libraries created by us
# Use this path for Windows
# sys.path.append('C:\Users\krm9c\Dropbox\Work_9292016\Research\Common_Libraries')
# sys.path.append('C:\Users\krm9c\Desktop\Research\Paper_1_codes')
# path= "E:\Research_Krishnan\Data\Data_case_study_1"
# For MAC or Unix use this Path
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Common_Libraries')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_1_codes')
# Set path for the data too
path            = "/Users/krishnanraghavan/Documents/Data-case-study-1"
myRollpath      = "/Users/krishnanraghavan/Documents/InfiniteRollData/"
myRespath       = "/Users/krishnanraghavan/Documents/Results"
# Now import everything
from Library_Paper_two       import *
from Library_Paper_one       import Collect_samples_Bearing
# Data importing routine
#########################################################################################
# Generate an infinite loop of rolling element data
def data_import(par, tree):
    if par == 0:
        # Rolling Elelment Bootstrap
        # Bootstrap sampling
        NL, IR, OR, Norm = Collect_samples_Bearing(path, 2000)
        R  = np.array(OR)
        R1 = np.array(NL)
        R2 = np.array(Norm)
        R3 = np.array(IR)
        T  = np.concatenate((R, R1, R2, R3))
        # Define the labels
        Y =  np.concatenate(((np.zeros((R.shape[0]))), (np.zeros((R1.shape[0]))+1), (np.zeros((R1.shape[0]))+2), (np.zeros((R3.shape[0]))+3) ))
        if tree ==0:
            return T, tflearn.data_utils.to_categorical(Y, Faults)
        else:
            # Pre-Train the Dimension Reducer
            Ref, Tree = initialize_calculation(T = None, Data = T, gsize = dimension, par_rbf = par, K_function = function_choose, par_train = 0)
            return Tree, T, tflearn.data_utils.to_categorical(Y, Faults)

def NN():
    # DATA
    Scalar = preprocessing.StandardScaler()
    T, Y = data_import(0,0);
    Scalar.fit(T)
    X = Scalar.transform(T)
    # Network Parameters
    n_hidden_1 = 1024 # 1st layer number of features
    n_hidden_2 = 256 # 2nd layer number of features
    n_input = 11 # MNIST data input (img shape: 28*28)
    n_classes = 4 # MNIST total classes (0-9 digits)
    # tf Graph input and label
    I = tf.placeholder("float", [None, T.shape[1]])
    L = tf.placeholder("float", [None, Faults])

    # Create model
    def multilayer_perceptron(I, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(I, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return tf.nn.softmax(out_layer)
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(I, weights, biases)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.square(pred - L))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Initializing the variables
    init = tf.initialize_all_variables()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(T.shape[0]/batch_size)
            prev=0
            end=prev+batch_size
            # Loop over all batches
            for i in range(total_batch):
                batch_x  = X[prev:end, :]
                batch_y  = Y[prev:end, :]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={I: batch_x,
                                                              L: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
                prev = end
                end = prev+batch_size
                # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Import Test Data
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(L, 1))
        Norm, IR, OR, NL = RollingDataImport(path, 1)
        R  = np.array(OR)
        R1 = np.array(NL)
        R2 = np.array(Norm)
        R3 = np.array(IR)
        P  = np.concatenate((R, R1, R2, R3))
        Scalar.fit(P)
        T = Scalar.transform(P)

        # Define the labels
        Y =  np.concatenate(((np.zeros((R.shape[0]))), (np.zeros((R1.shape[0]))+1), (np.zeros((R1.shape[0]))+2), (np.zeros((R3.shape[0]))+3) ))
        Y = tflearn.data_utils.to_categorical(Y, Faults)

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
        print("Accuracy:", accuracy.eval({I: T[0:400,:], L: Y[0:400,:]}))

        #Calculate Prediction probabilities
        prob     = tf.cast(tf.nn.softmax(pred),tf.float32)
        Xhat = T[0:T.shape[0],:]
        Yhat = Y[0:T.shape[0],:]
        _, Pred = sess.run([optimizer, prob], feed_dict = {I: Xhat, L: Yhat})
        print Pred
        np.savetxt("Prediction.csv", Pred, delimiter=',')

NN()
