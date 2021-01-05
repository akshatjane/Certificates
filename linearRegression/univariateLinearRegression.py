import numpy
import matplotlib.pyplot 
import pandas 
from mpl_toolkits.mplot3d import Axes3D

def computeCost(X,y,theta):
 m = len(y)
 predictions = X.dot(theta) 
 squareError = ( predictions - y ) ** 2
 cost =  1 / (2 * m)  * numpy.sum(squareError)
 return cost;

def gradientDescent(X,y,theta,alpha,numberOfIterations): 
 m = len(y);
 history = []
 for i in range(numberOfIterations):
  predictions = X.dot(theta)
  error = numpy.dot(X.transpose(),(predictions-y))
  descent = alpha * 1 / m * error 
  theta = theta - descent
  history.append(computeCost(X,y,theta))
 return theta,history;

def main():
 data = pandas.read_csv('ex1data1.txt',header=None)
 matplotlib.pyplot.scatter(data[0],data[1])
 matplotlib.pyplot.xlabel("Population Of City")
 matplotlib.pyplot.ylabel("Profit");
 matplotlib.pyplot.title("Profit Prediction")
 matplotlib.pyplot.show()
 dataValues = data.values;
 m = dataValues[:,0].size;
 X = numpy.append(numpy.ones((m,1)),dataValues[:,0].reshape(m,1),axis=1)
 y = dataValues[:,1].reshape(m,1) 
 theta = numpy.zeros((2,1))
 theta , history = gradientDescent(X,y,theta,0.01,1500)

if __name__ == '__main__':
 main();