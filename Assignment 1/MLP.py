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
        
    def fit(self, X_train, t_train, epochs = 100, diff=0.00001, momentum=0.2):
        """Initialize the weights. Train *epochs* many epochs."""
        # Scaling X_train correctly first
        self.momentum = momentum
        Xs_train = np.copy(X_train)
        x_max = np.max(Xs_train)
        x_min = np.min(Xs_train)
        Xs_train = (Xs_train - x_min)/(x_max - x_min) # scaled 

        Xs_train = np.concatenate((Xs_train,-np.ones((np.shape(Xs_train)[0],1))),axis=1)

        

        self.dim_in = X_train.shape[-1]

        self.dim_out = int(t_train.shape[-1])

        self.V = (np.random.rand(self.dim_in+1,self.dim_hidden)-0.5)*2/np.sqrt(self.dim_in)
        self.W = (np.random.rand(self.dim_hidden+1,self.dim_out)-0.5)*2/np.sqrt(self.dim_hidden)

        n_V = np.zeros((np.shape(self.V)))
        n_W = np.zeros((np.shape(self.W)))
        
        for e in range(epochs):

            self.forward(Xs_train)

            d_o = (self.Y - t_train)*self.Y*(1.0-self.Y)
            d_h = self.a*(1.0-self.a)*(np.dot(d_o,np.transpose(self.W)))
            n_V = self.eta*(np.dot(np.transpose(Xs_train),d_h[:,:-1])) + self.momentum*n_V
            n_W = self.eta*(np.dot(np.transpose(self.a),d_o)) + self.momentum*n_W
            self.V -= n_V
            self.W -= n_W



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
            h_o = np.dot(X, self.V)
            self.a = self.logistic(h_o)
            self.a = np.concatenate((self.a,-np.ones((np.shape(self.a)[0],1))),axis=1)
            h_k = np.dot(self.a, self.W)
            self.Y = self.logistic(h_k)
            #return Y
        #Fill in the code
        
    def confmat(self,inputs,targets, acc=0):
        # taken from marslands
        """Confusion matrix"""
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        #outputs = self.forward(inputs)
        self.forward(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            self.Y = np.where(self.Y>0.5,1,0)
        else:
            # 1-of-N encoding
            self.Y = np.argmax(self.Y,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(self.Y==i,1,0)*np.where(targets==j,1,0))

        if acc==0:
            print ("Confusion matrix is:")
            print (cm)
            print ("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)
        else:
            print ("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)
            return np.trace(cm)/np.sum(cm)*100



t_val = t_val.reshape((len(t_val), 1))
t_test = t_test.reshape((len(t_test), 1))
t_train = t_train.reshape((len(t_train), 1))
lr_cl =  MNNClassifier(eta = 0.01, dim_hidden=24)
lr_cl.fit(X_train, t_train, epochs=10000)
#print("predictions: ", lr_cl.predict(anddata[:,0:2]))
print(lr_cl.confmat(X_test, t_test))


"""
t_val = t_val.reshape((len(t_val), 1))
t_train = t_train.reshape((len(t_train), 1))
T_acc = []
V_acc = []
sets = np.arange(0, 1000, 20)
for e in sets:
    lr_cl =  MNNClassifier(eta = 0.01, dim_hidden=24)
    lr_cl.fit(X_train, M_train, epochs=e)
    T_acc.append(lr_cl.confmat(X_train, t_train, acc=1))
    V_acc.append(lr_cl.confmat(X_val, t_val, acc=1))

plt.plot(sets, np.asarray(T_acc), label="training accuracy")
plt.plot(sets, np.asarray(V_acc), label="validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy [%]")
plt.legend()
plt.show()
"""