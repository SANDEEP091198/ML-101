##AUTHOR: SANDEEP_REDDY_PERIYAVARAM
#------------------------------------------------------------------------------
##IMPORTING REQUIRED LIBRARIES
import numpy as np
#import pandas as pd
#from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#------------------------------------------------------------------------------
##CONFIGURING THE TRAINER
class poly:
    def __init__(self, X, Y, order, alpha, iterations):
        w = np.zeros((order+1,1))
        wtf = [tf.Variable(a) for a in w]
        self.input_ex = tf.constant(X)
        self.order = order
        self.weights = wtf
        self.alpha = alpha
        self.iter = iterations
        self.output = tf.constant(Y)

    def function(self, X):
        res = np.ones(len(X)) * self.weights[-1]
        for i in range(self.order):
            res = res + ((X**(i+1))*(self.weights[i]))
        return res
  
    def loss(self, predicted_Y, Y):
        return tf.reduce_mean(tf.square(predicted_Y - Y))
    
    def train(self):
        for n in range(self.iter):
            with tf.GradientTape() as tape:
                hx = self.function(self.input_ex)
                current_loss = self.loss(hx, self.output)
                
                if n == 0:
                    print('Iteration: ', n+1)
                    print('Loss: ', current_loss)
                elif (n+1) % 100 == 0:
                    print('Iteration: ', n+1)
                    print('Loss: ', current_loss)
                
                gradients = tape.gradient(current_loss, self.weights)
                for i in range(self.order):
                    self.weights[i].assign_sub(self.alpha * gradients[i])
                self.weights[-1].assign_sub(self.alpha * gradients[-1])
        return self.weights
#------------------------------------------------------------------------------
##TESTING
#GENERATING RANDOM DATA
X = np.random.random((20, 1))
X = np.sort(X, 0)
Y = ((X**3) * 3) + ((X**2) * 2) + (X * 1) + 0.5
order = 3

#DECLARING HYPERPARAMETERS
alpha = 0.01
iterations = 1000

#TRAINING THE MODEL
trainer = poly(X, Y, order, alpha, iterations)
weights = trainer.train()
w = weights[0:order]
b = weights[-1]

#PREDICTING OUTPUT
val = -0.25
p = (val**3)*w[2] + (val**2)*w[1] + (val**1)*w[0] + b
#------------------------------------------------------------------------------
##PLOTTING THE POLYNOMIAL FUNCTION
def linear_plot(X, Y):
    plt.scatter(X, Y, c = 'green')
    plt.scatter(val, p, c = 'red')
    T = trainer.function(X)
    plt.plot(X, T)
    plt.grid()
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Data Distribution')
    plt.show()

linear_plot(X, Y)
#------------------------------------------------------------------------------
