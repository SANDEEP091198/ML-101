##AUTHOR: SANDEEP_REDDY_PERIYAVARAM
#------------------------------------------------------------------------------
##IMPORTING REQUIRED LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
##CONFIGURING THE TRAINER
class reg:
    #INSTANTIATING THE PARAMETERS
    def __init__(self, X, Y, alpha, iterations):
        ymin = np.min(Y)
        ymax = np.max(Y)
        imin = np.where(Y == ymin)[0][0]
        imax = np.where(Y == ymax)[0][0]
        self.x_coeff = (Y[imax] - Y[imin])/(X[imax] - X[imin])
        self.constant = 0
        self.alpha = alpha
        self.iter = iterations
    
    #DEFINING THE LINEAR FUNCTION
    def function(self, X, w, b):
        res = (X*w) + b
        return res
    
    #MINIMIZING THE LOSS
    def train_model(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        
        def cost(n):
            hx = self.function(X, self.x_coeff, self.constant)
            j = sum((Y-hx)**2)/(2*len(X))
            dw = -sum(np.multiply((Y-hx), X))/len(X)
            db = -sum(Y-hx)/len(X)
            if n == 0:
                print('Cost: ', j)
            elif (n+1) % 10 == 0:
                print('Cost: ', j)
            return dw, db
        
        for n in range(self.iter):
            if n == 0:
                print('Iteration: ', n+1)
            elif (n+1) % 10 == 0:
                print('Iteration: ', n+1)
            dw, db = cost(n)
            self.x_coeff -= alpha*(dw)
            self.constant -= alpha*(db)

        return self.x_coeff, self.constant
    
    #PREDICTING ON NEW INPUT
    def predict(self, val):
        res = self.function(val, self.x_coeff, self.constant)
        return res
#------------------------------------------------------------------------------
##TESTING
#GENERATING RANDOM DATA
X = 30 * np.random.random((20, 1))
Y = (X * 0.5) + 1.0 + np.random.normal(size=X.shape)

#DECLARING HYPERPARAMETERS
alpha = 0.001
iterations = 50

#TRAINING THE MODEL
trainer = reg(X, Y, alpha, iterations)
w_trained, b_trained = trainer.train_model(X, Y)
val = 23.625
p = trainer.predict(val)
print(p)
#------------------------------------------------------------------------------
##PLOTTING THE LINEAR FUNCTION
def linear_plot(X, Y, w_trained, b_trained):
    plt.scatter(X, Y, c = 'green')
    plt.scatter(val, p, c = 'red')
    T = (X * w_trained) + b_trained
    plt.plot(X, T)
    plt.grid()
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Data Distribution')
    plt.show()

linear_plot(X, Y, w_trained, b_trained)
#------------------------------------------------------------------------------