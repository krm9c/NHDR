# Time Series Simulation
# Author : Krishnan Raghavan
# Date: Dec 25, 2016
#######################################################################################
# Define all the libraries
import os, sys, random, time, tflearn
import numpy as np
from   sklearn import preprocessing
import tensorflow as tf
#######################################################################################
# Libraries created by us
# Use this path for Windows
# sys.path.append('C:\Users\krm9c\Dropbox\Work_9292016\Research\Common_Libraries')
# sys.path.append('C:\Users\krm9c\Desktop\Research\Paper_1_codes')
# path= "E:\Research_Krishnan\Data\Data_case_study_1"
# For MAC or Unix use this Path
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Common_Libraries')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_1_codes')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_2_codes')

# Set path for the data too
path = "/Users/krishnanraghavan/Documents/Data-case-study-1"
myRespath = "/Users/krishnanraghavan/Dropbox/Work/Research/Paper-2-Codes/Results"
# Now import everything
from Library_Paper_two import *
from Library_Paper_one import Collect_samples_Bearing
###################################################################################
# Setup some parameters for the analysis
# Some global parameters for the script
# The NN parameters
Train_batch_size          = 100
Test_batch_size           = 100
Train_Glob_Iterations     = 10
Train_Loc_Iterations      = 1000
Deviation_par 			  = 0.9
# Start a time clock
start_time                = time.time()

# Set up parameters for dimension reduction
Faults                    = 4
dimension                 = 4
function_choose           = "linear"
par                       = 0
###################################################################################
# Create a infinite Loop of Data-stream
def RollingDataImport(path, num):
	# Start_Analysis_Bearing()
	IR =  np.loadtxt('Results/Data/IR_sample.csv', delimiter=',')
	OR =  np.loadtxt('Results/Data/OR_sample.csv', delimiter=',')
	NL =  np.loadtxt('Results/Data/NL_sample.csv', delimiter=',')
	Norm =  np.loadtxt('Results/Data/Norm.csv'     , delimiter=',')
	sheet    = 'Test';
	f        = 'IR'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_IR  =  np.array(import_data(filename,sheet, 1));
	sheet    = 'Test';
	f        = 'OR'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_OR  =  np.array(import_data(filename,sheet, 1));
	sheet    = 'Test';
	f        = 'NL'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_NL  =  np.array(import_data(filename,sheet, 1));
	sheet    = 'normal';
	f        = 'Normal_1.xls'
	filename = os.path.join(path,f);
	Temp_Norm= np.array(import_data(filename,sheet, 1));
	return Temp_Norm, Temp_IR, Temp_OR, Temp_NL

###################################################################################
# Global Import of Data
Norm, IR, OR, NL = RollingDataImport(path, 1)
Scalar = preprocessing.StandardScaler(with_mean = False)
R  = np.array(OR)
R1 = np.array(NL)
R2 = np.array(Norm)
R3 = np.array(IR)
P  = np.concatenate((R, R1, R2, R3))
Y  = np.concatenate(((np.zeros((R.shape[0]))), (np.zeros((R1.shape[0]))+1), (np.zeros((R2.shape[0]))+2), (np.zeros((R3.shape[0]))+3) ))
Scalar.fit(P)
T = Scalar.transform(P)
def Inf_Loop_data(par):
	if par ==1:
		Ref, Tree = initialize_calculation(T = None, Data = T, gsize = dimension, par_rbf = par, K_function = function_choose, par_train = 0)
		return (T + np.random.normal(0, 0.1, (T.shape[0], T.shape[1]))), Tree, tflearn.data_utils.to_categorical(Y, Faults)
	else:
		return (T + np.random.normal(0, 0.1, (T.shape[0], T.shape[1]))), tflearn.data_utils.to_categorical(Y, Faults)
####################################################################################
# Helper Function for the weight and the bias variable
# Tensorflow mahalanobis distance
def tensorflow_MTS(D1, D2 , Faults, model):
	det = tf.pow(Deviation_par, Faults)
	Sigma = tf.multiply( tf.constant( [1.0] ) , tf.diag( tf.ones( [Faults] ) ) )
	# Declare the two matrices
	Ehat =  tf.matmul( tf.matrix_inverse( tf.sqrt( Sigma ) ), D2, transpose_b = True )
	# Defining the cost function
	return (tf.exp( tf.neg( tf.add(  ( ((Faults/float(2))*log(2*pi)) +  ((1/float(2)) * det) ) , tf.multiply((1/float(2)), (tf.square(Ehat[0]) + tf.square(Ehat[1])+ tf.square(Ehat[2])+ tf.square(Ehat[3])))  )  )  ) )
# Weight
def weight_variable(shape, trainable, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, trainable = trainable, name = name)
# Bias function
def bias_variable(shape, trainable, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, trainable = trainable, name = name)
#  Summaries for the variables
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram_1', var)
# Class
class Agent():
    def __init__(self):
		self.classifier = {}
		self.Deep = {}
		self.Trainer = {}
		self.Summaries = {}
		self.sess = tf.InteractiveSession()

    # Function for defining every NN
    def nn_layer(self, input_tensor, input_dim, output_dim, act, trainability, key):
        with tf.name_scope(key):
            with tf.name_scope('weights'+key):
                self.classifier['Weight'+key] = weight_variable([input_dim, output_dim], trainable = trainability, name = 'Weight'+key)
                variable_summaries(self.classifier['Weight'+key])
            with tf.name_scope('bias'+key):
                self.classifier['Bias'+key] = bias_variable([output_dim], trainable = trainability, name = 'Bias'+key)
                variable_summaries(self.classifier['Weight'+key])
            with tf.name_scope('Wx_plus_b'+key):
                preactivate = tf.matmul(input_tensor, self.classifier['Weight'+key]) + self.classifier['Bias'+key]
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation'+key)
            tf.summary.histogram('activations', activations)
        return activations

    # Initialization for the default graph and the corresponding NN.
    def init_NN(self, R, Faults, lr, depth, Layers):
		Keys_Weight = []
		Keys_Bias =[]
		List_Weight = []
		List_Bias =[]
		with tf.name_scope("FLearners"):
			self.Deep['FL_layer0'] = tf.placeholder(tf.float32, shape=[None, R.shape[1]])
        	for i in range(1,len(Layers)):
				self.Deep['FL_layer'+str(i)] = self.nn_layer(self.Deep['FL_layer'+str(i-1)], Layers[i-1], Layers[i], act=tf.nn.relu, trainability = False, key = 'FL_layer'+str(i))
				Keys_Weight.append(self.classifier['Weight'+'FL_layer'+str(i)])
				Keys_Bias.append(self.classifier['Bias'+'FL_layer'+str(i)])
		with tf.name_scope("dropout"):
			self.Deep['keep_prob'] = tf.placeholder(tf.float32)
			self.Deep['layer_dropout'] = tf.nn.dropout(self.Deep['FL_layer'+str(len(Layers)-1)], self.Deep['keep_prob'])
		with tf.name_scope("Targets"):
			self.classifier['Target'] = tf.placeholder(tf.float32, shape=[None, Faults])
		with tf.name_scope("Classifier"):
			o = []
			for i in range(Faults):
				self.classifier['Fault'+str(i)] = self.nn_layer( self.Deep['layer_dropout'], Layers[(len(Layers)-1)], 1, act=tf.identity, trainability =  False, key = 'Fault'+str(i))
				o.append(self.classifier['Fault'+str(i)])
				List_Weight.append(self.classifier['WeightFault'+str(i)])
				List_Bias.append(self.classifier['BiasFault'+str(i)])
    		o = (tf.transpose(tf.squeeze(o)))
    		tf.summary.histogram('Output', o)

		with tf.name_scope("Trainer"):

			# Define the training routine
			E = tf.sub( self.classifier['Target'], o )
			Sigma = tf.multiply( tf.constant( [1.0] ) , tf.diag( tf.ones( [Faults] ) ) )

			# Declare the two matrices
			Ehat =  tf.matmul( tf.matrix_inverse( tf.sqrt( Sigma ) ), E, transpose_b = True )
			# Regularization terms
			# reg_const = 0
			# lossL21   = tf.add_n([tf.nn.l2_loss(v) for v in Keys_Weight])
			# lossL22   = tf.add_n([tf.nn.l2_loss(v) for v in List_Weight])
			# Declare a temporary cost value
			temp_c 	  = tf.square(Ehat[0]) + tf.square(Ehat[1]) + tf.square(Ehat[2]) + tf.square(Ehat[3])
			# Defint the two losses
			Loss =  tf.multiply( tf.constant([(1/float(2))]), temp_c)
			# Defining the cost function
			self.Trainer["cost"]= (tf.reduce_mean( Loss ) )# + reg_const*( lossL22+lossL21 ))
			tf.summary.scalar('Cost', self.Trainer["cost"])
			global_step = tf.Variable(0, trainable=False)
			learning_rate = tf.train.exponential_decay( lr, global_step, 1000, 0.99, staircase=True )
			tf.summary.scalar('LearningRate', learning_rate)
			self.Trainer['Optimizer']    =  tf.train.AdamOptimizer(learning_rate)
			for i in range(Faults):
				var_list = Keys_Weight+[self.classifier['WeightFault'+str(i)], self.classifier['BiasFault'+str(i)]]
				self.Trainer["TrainStep"+str(i)] =self.Trainer['Optimizer'].minimize(self.Trainer["cost"], var_list = var_list, global_step = global_step)
			with tf.name_scope('Evaluation'):
				with tf.name_scope('CorrectPrediction'):
					self.Trainer['correct_prediction'] = tf.equal(tf.argmax(self.classifier['Target'],1), tf.argmax(o,1))
				with tf.name_scope('Accuracy'):
					self.Trainer['accuracy'] = tf.reduce_mean(tf.cast(self.Trainer['correct_prediction'], tf.float32))
				with tf.name_scope('Prob'):
					probs = tensorflow_MTS(self.classifier['Target'], o, Faults, self)
					self.Trainer['prob'] = tf.cast(probs,tf.float32)
				tf.summary.scalar('Accuracy', self.Trainer['accuracy'])
				tf.summary.histogram('Prob', self.Trainer['prob'])
		self.Summaries['merged'] = tf.summary.merge_all()
		self.Summaries['train_writer'] = tf.summary.FileWriter(myRespath + '/train', self.sess.graph)
		self.Summaries['test_writer'] = tf.summary.FileWriter(myRespath + '/test')
		self.sess.run(tf.global_variables_initializer())
		List = List_Weight+ List_Bias
		Keys = Keys_Weight+ Keys_Bias
		return self, List, Keys

#######################################################################
# Main Training Function for the model
def Training():
	# Generate the data and define parameter
	T, Y =  Inf_Loop_data(0)
	# Lets start with creating a model and then train batch wise.
	model= Agent()
	model, var_list_classifier, var_list_FLearner = model.init_NN(T, Faults, 1e-3, 2, [T.shape[1], 1024])
	print "Everythin Declared, I am getting into the training loop"
	# Declare a saver
	for i in range(Train_Glob_Iterations):
		X, Y = Inf_Loop_data(0)
		for j in range(Train_Loc_Iterations):
			prev=0
			end=prev+Train_batch_size
			flag = 1
			for k in range(((X.shape[0])/Train_batch_size)):
				batch_xs  = X[prev:end, :]
				batch_ys  = Y[prev:end, :]
				for l in range(Faults):
					summary, _  = model.sess.run([model.Summaries['merged'], model.Trainer['TrainStep'+str(l)]], feed_dict ={model.Deep['FL_layer0'] : batch_xs, model.classifier['Target']: batch_ys, model.Deep['keep_prob']: 0.4})
					if j % 10 == 0 and flag < 4 :
						model.Summaries['train_writer'].add_summary(summary, j)
						flag = flag+1
				prev = end
				end  = prev+Train_batch_size

		if i % 2 == 0:
			T, Y = Inf_Loop_data(0)
			summary, acc  = model.sess.run([model.Summaries['merged'], model.Trainer['accuracy']],feed_dict ={model.Deep['FL_layer0'] : T, model.classifier['Target']: Y, model.Deep['keep_prob']: 1} )

			summary, prob  = model.sess.run([model.Summaries['merged'], model.Trainer['prob']],feed_dict ={model.Deep['FL_layer0'] : T, model.classifier['Target']: Y, model.Deep['keep_prob']: 1} )
			np.savetxt('prob'+str(i)+'.csv', np.array(prob), delimiter = ',')

			print( "Iteration == ", i, "test accuracy ==%g"%acc)
			model.Summaries['test_writer'].add_summary(summary, i)
if __name__ == "__main__":
	Training()
