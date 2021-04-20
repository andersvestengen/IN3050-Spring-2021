import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
import random


from sklearn.datasets import make_blobs
X, t = make_blobs(n_samples=[400,800,400], centers=[[0,0],[1,2],[2,3]], 
                  n_features=2, random_state=2019)

def add_bias(X):
    # Put bias in position 0
    sh = X.shape
    if len(sh) == 1:
        #X is a vector
        return np.concatenate([np.array([1]), X])
    else:
        # X is a matrix
        m = sh[0]
        bias = np.ones((m,1)) # Makes a m*1 matrix of 1-s
        return np.concatenate([bias, X], axis  = 1) 

indices = np.arange(X.shape[0])
random.seed(2020)
random.shuffle(indices)
indices[:10]

X_train = X[indices[:800],:]
X_val = X[indices[800:1200],:]
X_test = X[indices[1200:],:]
t_train = t[indices[:800]]
t_val = t[indices[800:1200]]
t_test = t[indices[1200:]]

t2_train = t_train == 1
t2_train = t2_train.astype('int')
t2_val = (t_val == 1).astype('int')
t2_test = (t_test == 1).astype('int')


m_t0_train = t_train == 0
m_t0_train = m_t0_train.astype('int')
m_t0_val = (t_val == 0).astype('int')
m_t0_test = (t_test == 0).astype('int')

m_t1_train = t_train == 1
m_t1_train = m_t1_train.astype('int')
m_t1_val = (t_val == 1).astype('int')
m_t1_test = (t_test == 1).astype('int')

m_t2_train = t_train == 2
m_t2_train = m_t2_train.astype('int')
m_t2_val = (t_val == 2).astype('int')
m_t2_test = (t_test == 2).astype('int')

M_train = np.zeros((800, 3))
M_train[:,0] = m_t0_train
M_train[:,1] = m_t1_train
M_train[:,2] = m_t2_train


class MNNClassifier():
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self,eta = 0.001, dim_hidden = 6):
        """Initialize the hyperparameters"""
        self.eta = eta
        self.dim_hidden = dim_hidden

        # Should you put additional code here?
        
    def fit(self, X_train, t_train, epochs = 100, diff=0.00001):
        """Initialize the weights. Train *epochs* many epochs."""
        # Scaling X_train correctly first
        Xs_train = np.copy(X_train)
        x_max = np.max(Xs_train)
        x_min = np.min(Xs_train)
        Xs_train = (Xs_train - x_min)/(x_max - x_min) # scaled 
        #print("The input and X training shape:", Xs_train.shape)
        Xs_train = add_bias(Xs_train)
        #print("The input and X training shape:", Xs_train.shape)
        
        # Initializing the weights
        dim_in = X_train.shape[-1]
        #dim_out = int(len(np.unique(t_train)))
        dim_out = int(t_train.shape[-1])
        #print("This is dim_in:", dim_in)
        #print("This is dim_out:", dim_out)
        V = np.random.rand(dim_in+1, self.dim_hidden)
        W = np.random.rand(self.dim_hidden+1, dim_out)
        #change = np.asarray([0, 1, 2])
        
        for e in range(epochs):

            #Forward - phase
            
            #print("This is the V shape:", V.shape)
            #print("This is the W shape:", W.shape)
            h_o = np.dot(Xs_train, V)
            #print("The shape of h_o activation function:", h_o.shape)
            a = self.logistic(h_o)
            a = self.add_bias(a)
            #print("This is the output of activation func a:", a.shape)
            h_k = np.dot(a, W)
            #print("this is the h_k shape:", h_k.shape)
            Y = self.logistic(h_k)
            #print("This is the output Y shape:", Y.shape)
            
            #Backward - phase
            
            d_o = (t_train - Y)*Y*(1.0-Y)
            #print("This is the d_0 shape:", d_o.shape)
            d_h = a*(1-a)*np.dot(d_o, np.transpose(W))
            #print("this is the d_h shape:", d_h.shape)
            n_V = self.eta*(np.dot(np.transpose(Xs_train), d_h[:,:-1]))
            #print("this is n_V shape:", n_V.shape)
            n_W = self.eta*(np.dot(np.transpose(a), d_o))
            #print("this is n_W shape:", n_W.shape)
            if np.linalg.norm(d_o) < diff:
                print("Hit diff!")
                break
            V += n_V
            W += n_W
            # shuffle the inputs
            #np.random.shuffle(change)
            #Xs_train = Xs_train[:,change]
            #t_train = t_train[:,change]
            #print("this is d_o", np.linalg.norm(d_o))
            
        self.weights1 = V
        self.weights2 = W
        #print("This is final V", V)
        #print("This is empty space")
        #print("this is final W", W)

    def logistic(self, x):
        return 1/(1+np.exp(-x))       
        
        
    def add_bias(self, X):
        # Put bias in position 0
        sh = X.shape
        if len(sh) == 1:
            #X is a vector
            return np.concatenate([np.array([1]), X])
        else:
            # X is a matrix
            m = sh[0]
            bias = np.ones((m,1)) # Makes a m*1 matrix of 1-s
            return np.concatenate([bias, X], axis  = 1) 
            
    def forward(self, X):
            X = self.add_bias(X)
            h_o = np.dot(X, self.weights1)
            a = self.logistic(h_o)
            a = self.add_bias(a)
            h_k = np.dot(a, self.weights2)
            Y = self.logistic(h_k)
            return Y
        #Fill in the code

    def predict(self, x, threshold=0.5):
        predictions = []
        for i in self.forward(x):
            predictions.append(np.argmax(i))
        predictions = np.asarray(predictions).astype('int')
        
        return predictions
    
    #def accuracy(self, X_test, t_test):
    #    """Calculate the accuracy of the classifier for the pair (X_test, t_test)
    #    Return the accuracy"""
    #    #Fill in the code
        
    def accuracy(self,X_test, y_test, **kwargs):
        pred = self.predict(X_test, **kwargs)
        if len(pred.shape) > 1:
            pred = pred[:,0]
        return sum(pred==y_test)/len(pred)

T_acc = []
V_acc = []
sets = np.arange(0, 1000, 10)
for e in sets:
    lr_cl =  MNNClassifier(eta = 0.01, dim_hidden=24)
    lr_cl.fit(X_train, M_train, epochs=e)
    T_acc.append(lr_cl.accuracy(X_train, t_train))
    V_acc.append(lr_cl.accuracy(X_val, t_val))

plt.plot(sets, np.asarray(T_acc)*100, label="training accuracy")
plt.plot(sets, np.asarray(V_acc)*100, label="validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy [%]")
plt.legend()
plt.show()
