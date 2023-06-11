# ML-101

The following Machine Learning models can be imported as custom modules by following the steps provided below:

    I.    Download the required code file
    
    II.   Comment out all the lines starting from '##TESTING'
    
    III.  Add the file location to sys.path
    
    IV.   Invoke 'from <filename> import *'
    
    V.    Prepare and analyze your data
    
    VI.   Train the model as shown in the '##TESTING' section
    
    VII.  [Optional] Plot the datapoints as shown in '##PLOTTING THE LINEAR FUNCTION' section

NOTE: In case the algorithm fails to converge, try normalizing the data for better performance.

## LINEAR_REGRESSION

This is a simple Linear Regression model with mean squared error loss.

To improve the training efficiency, x-coefficient has been initiated by computing the slope of the line joining the two extreme points on the training set.

## POLYNOMIAL_REGRESSION

This is a generalized model that accepts user input for the order of the polynomial function.

Additionally, gradient descent is achieved through auto-differentiation using tensorflow.

However, in dealing with higher order input features in real-life applications, Neural Networks are recommended.
