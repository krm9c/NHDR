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


# Some global parameters for the script
Train_batch_size          = 50
Test_batch_size           = 10
start_time                = time.time()
Faults                    = 4
dimension                 = 4
function_choose           = "linear"
par                       = 0
Test_Iteration            = 10
Train_Glob_Iterations     = 20
Train_Loc_Iterations      = 100

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
# Class for the Fault classification Network
#######################################################################################
# Helper Function
def weight_variable(shape, trainable, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, trainable = trainable, name = name)

#######################################################################################
def bias_variable(shape, trainable, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, trainable = trainable, name = name)

#######################################################################################
class Agent():
    def __init__(self, dim):
        self.input      = tf.placeholder(tf.float32, shape=[None, dim])
        self.classifier = {}
        self.weights    = {}
        self.bias       = {}
        self.Trainer    = {}
        self.Deep       = {}
        self.Output     = {}
        self.sess       = tf.InteractiveSession()
        self.merged     = 0
        self.keep_prob  = 0

    def nn_layer(self, input_tensor, input_dim, output_dim, act, trainability, name):
        self.weights['Weight'+str(name)] = weight_variable([input_dim, output_dim], trainable = trainability, name = 'Weight'+str(name))
        self.bias['Bias'+str(name)] = bias_variable([output_dim], trainable = trainability, name = 'Bias'+str(name))
        preactivate = tf.matmul(input_tensor, self.weights['Weight'+str(name)]) + self.bias['Bias'+str(name)]
        activations = act(preactivate, name='activation'+name)
        return activations

    def Init_NN(self, R, Faults, lr, depth, Layers):
        print "The number of Faults are ", Faults
        self.Deep['layer0']    = self.input
        for i in range(1,len(Layers)):
            self.Deep['layer'+str(i)] = self.nn_layer(self.Deep['layer'+str(i-1)], Layers[i-1], Layers[i], act=tf.nn.relu, trainability = True, name = 'layer'+str(i))
        self.keep_prob                = tf.placeholder(tf.float32)
        self.Deep['layer_dropout']    = tf.nn.dropout(self.Deep['layer'+str(len(Layers)-1)], self.keep_prob)
        self.Output['Target']         = tf.placeholder(tf.float32, shape=[None, Faults])
        o = []
        for i in range(Faults):
            self.classifier['Fault'+str(i)] = self.nn_layer( self.Deep['layer_dropout'], Layers[(len(Layers)-1)], 1, act=tf.sigmoid, trainability =  False, name = 'Fault'+str(i))
            o.append(self.classifier['Fault'+str(i)])
        o = (tf.transpose(tf.squeeze(o)))
        self.Trainer["cost"] = tf.reduce_mean(tf.cast(tf.square(o[0]-self.Output['Target'][0])+tf.square(o[1]-self.Output['Target'][1])+tf.square(o[2]-self.Output['Target'][2])+tf.square(o[3]-self.Output['Target'][3]), tf.float32))
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, 100000, 0.99, staircase=True)
        self.Trainer['Optimizer']    =  tf.train.AdamOptimizer(learning_rate)
        for i in range(Faults):
            self.Trainer["TrainStep"+str(i)] =self.Trainer['Optimizer'].minimize(self.Trainer["cost"], var_list = [self.weights['WeightFault'+str(i)], self.bias['BiasFault'+str(i) ]], global_step = global_step)
        # The accuracy estimation routine for the data.
        self.Trainer['correct_prediction'] = tf.equal(tf.argmax(self.Output['Target'],1), tf.argmax(o,1))
        self.Trainer['accuracy']           = tf.reduce_mean(tf.cast(self.Trainer['correct_prediction'], tf.float32),0)
        self.Trainer['prob']               = tf.reduce_mean(tf.cast((o),tf.float32),0)
        self.sess.run(tf.initialize_all_variables())
        return self

#######################################################################
def Test(model, Tree):
    # Batch training Routine
    Progression     = []
    Fault_Point     = []
    Outlier         = []
    Scalar = preprocessing.StandardScaler()
    from sklearn.utils import shuffle
    # Testing routine
    for i in range(Test_Iteration):

        # Let us get some data
        Norm, IR, OR, NL = RollingDataImport(path, 1)
        R  = np.array(Norm)
        R1 = np.array(IR)
        R2 = np.array(OR)
        R3 = np.array(NL)
        # P  = np.concatenate((R, R1, R2, R3))
        P = R1
        Scalar.fit(P)
        T = Scalar.transform(P)
        # Define the labels
        # Y =  np.concatenate(( (np.zeros((R.shape[0]))), (np.zeros((R1.shape[0]))+1), (np.zeros((R2.shape[0]))+2), (np.zeros((R3.shape[0]))+3) ))
        Y = (np.zeros((P.shape[0]))+1)
        Y = tflearn.data_utils.to_categorical(Y, Faults)
        # X, Tree =  initialize_calculation(T = Tree, Data= T, gsize  =dimension, par_rbf = par, K_function=function_choose, par_train= 1)
        Scalar.fit(T)
        X = Scalar.transform(T)
        # X, Y = shuffle(X, Y, random_state=0)
        prev=0
        end=prev+Test_batch_size
        for j in range(((X.shape[0])/Test_batch_size)):
            batch_xs  = X[prev:end, :]
            batch_ys  = Y[prev:end, :]
            Temp  =  model.Trainer['prob'].eval(feed_dict ={model.input : batch_xs, model.Output['Target']: batch_ys.reshape([batch_ys.shape[0],Faults]), model.keep_prob: 1})
            winner = np.max(Temp)
            # Three cases
            thresh_1 = 0.7
            thresh_2 = 0.2
            # Case -- 1
            if winner >  thresh_1:
                Fault_Point.append([winner, i, j, np.argmax(Temp)])
                for i in range(Faults):
                    if i is not np.argmax(Temp):
                        model.Trainer['TrainStep'+str(i)].run(feed_dict ={model.input : batch_xs, model.Output['Target']: batch_ys, model.keep_prob: 0.5})
            elif winner < thresh_2:
            # Case -- 2
                Outlier.append([winner, i, j, np.argmax(Temp)])
            else:
            # Case -- 3
                Progression.append([winner, i, j, np.argmax(Temp)])
            prev = end
            end  = prev+Test_batch_size
        print "##### i ==", i, "#####"
        print "Number of Faulty", len(Fault_Point)
        print "Number of Progression", len(Progression)
        print "Number of Outlier", len(Outlier)
        Fault_Point = []
        Progression = []
        print X.shape
        print Y.shape
        print("Test Accuracy ==%g"%model.Trainer['accuracy'].eval(feed_dict ={model.input : X, model.Output['Target']: Y.reshape([Y.shape[0],Faults]), model.keep_prob: 1}))
#######################################################################
# Main Training Function for the model
def Training():
    # Generate the data and define parameter
    Tree, T, Y =  data_import(0, 1)
    # Dimension Reduce the data
    # X, Tree =  initialize_calculation(T = Tree, Data= T, gsize  =dimension, par_rbf = par,   K_function=function_choose, par_train= 1)
    # Split the data for training purposes
    from sklearn.model_selection import train_test_split
    Scalar = preprocessing.StandardScaler()
    Scalar.fit(T)
    X = Scalar.transform(T)

    # Lets start with creating a model and then train batch wise.
    model= Agent(X.shape[1])
    model.Init_NN(X, Faults, 1e-4, 2, [X.shape[1], 1024])
    Pert = 0
    for i in range(Train_Glob_Iterations):
        T, Y = data_import(0, 0)
        # X, Tree =  initialize_calculation( T = Tree, Data= T, gsize  =dimension, par_rbf = par,   K_function=function_choose, par_train= 1 )
        Scalar.fit(T)
        X = Scalar.transform(T)
        for j in range(Train_Loc_Iterations):
            prev=0
            end=prev+Train_batch_size
            for j in range(((X.shape[0])/Train_batch_size)):
                batch_xs  = X[prev:end, :]
                batch_ys  = Y[prev:end, :]
                for k in range(Faults):
                    model.Trainer['TrainStep'+str(k)].run(feed_dict ={model.input : batch_xs, model.Output['Target']: batch_ys.reshape([Train_batch_size,Faults]), model.keep_prob: 0.7})
                prev = end
                end  = prev+Train_batch_size
        if i % 1 == 0:
            T, Y = data_import(0, 0)
            print( "Iteration == ", i, "test accuracy ==%g"%model.Trainer['accuracy'].eval(feed_dict ={model.input : X, model.Output['Target']: Y.reshape([Y.shape[0],Faults]), model.keep_prob: 1}))
    Test(model, Tree)
if __name__ == "__main__":
    Training()
