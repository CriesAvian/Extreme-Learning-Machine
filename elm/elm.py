import numpy as np
from scipy.linalg import pinv2

class elm:
  def __init__(self, architecture, activation, class_type):
    self.architecture = architecture
    self.activation = activation
    self.class_type = class_type
    self.input_weights = list()
    self.B = list()

  def relu(self, x):
    return np.maximum(x, 0, x)

  def sigmoid(self, x):
      s=1/(1+np.exp(-x))
      ds=s*(1-s)  
      return s

  def tanh(self, x):
      t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
      dt=1-t**2
      return t

  def softmax(self, x):
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum()

  def neuron_activation(self, activation, d):
      if activation == 'relu':
        H = self.relu(d)

      if activation == 'sigmoid':
        H = self.sigmoid(d)
      
      if activation == 'tanh':
        H = self.tanh(d)
      
      if activation == 'softmax':
        H = self.softmax(d)

      return H

  #Train
  def fit(self,X,y):
    input_weights = []
    # y = y.reshape(-1,1)
    # print(self.architecture)

    for layer in range(np.asarray(self.architecture).shape[0]):
      input_size = X.shape[1]
      self.input_weights.append(np.random.normal(size=[self.architecture[layer],input_size]))
      H = np.dot(self.input_weights[layer],X.T)
      
      self.neuron_activation(self.activation,H)

      X = H.T

    self.B = np.dot(pinv2(H).T,y)

  #Prediction
  def predict(self,X):
    for layer in range(np.asarray(self.architecture).shape[0]):
      input_size = X.shape[1]
      H = np.dot(self.input_weights[layer],X.T)
      
      self.neuron_activation(self.activation,H)
      
      X = H.T
    
    #Classification Task
    if self.class_type == 0:
      output = np.round(np.dot(H.T,self.B))
      return output
    
    #Regression Task
    if self.class_type == 1:
      output = np.dot(H.T,self.B)
      return output
