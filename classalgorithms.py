import numpy as np

import MLCourse.utilities as utils

# Susy: ~50 error
class Classifier:
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = parameters
        self.min = 0
        self.max = 1

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the training data """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        ytest = utils.threshold_probs(probs)
        return ytest



# Susy: ~27 error
class LinearRegressionClass(Classifier):
    def __init__(self, parameters = {}):
        self.params = {'regwgt': 0.01}
        self.weights = None
    
    def learn(self, X, y):
        # Ensure y is {-1,1}
        y = np.copy(y)
        y[y == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = X.shape[0]
        numfeatures = X.shape[1]

        inner = (X.T.dot(X) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(X.T).dot(y) / numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest
    

#Ans to the Q.no - 1(a)
# Susy: ~25 error
class NaiveBayes(Classifier):
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = utils.update_dictionary_items({'usecolumnones': False}, parameters)


    def learn(self, Xtrain, ytrain):
        # obtain number of classes
        if ytrain.shape[1] == 1:
            self.numclasses = 2
        else:
            raise Exception('Can only handle binary classification')
        
        #obtain number of features
        self.numfeatures=Xtrain.shape[1]
        
        #discarding column of ones in the predictor
        if self.params['usecolumnones'] == False:
            Xtrain=np.delete(Xtrain, -1, axis=1)
            
        self.numfeatures=Xtrain.shape[1]
        origin_shape = (self.numclasses,self.numfeatures)
        self.means= np.zeros(origin_shape)
        self.stds= np.zeros(origin_shape)
        
        #keeping count of the number of class=0 and class=1
        num_C_0=0
        num_C_1=0
        
        #based on the output y, separating two inputs class
        for i in range(len(ytrain)):
            if ytrain[i] == 0.0:
                num_C_0+=1
                self.means[0] = self.means[0]+ Xtrain[i][:self.numfeatures]
            if ytrain[i] == 1.0:
                num_C_1+=1
                self.means[1] = self.means[1]+ Xtrain[i][:self.numfeatures]
        
        
        #calculating the prior
        self.p_0=num_C_0/len(ytrain)
        self.p_1=num_C_1/len(ytrain)
        
        #calculating the mean
        for i in range(self.numfeatures):
            self.means[0][i]=self.means[0][i]/num_C_0
            self.means[1][i]=self.means[1][i]/num_C_1
        
        #calculating the standard deviation
        for i in range(len(ytrain)):
            if ytrain[i] == 0.0:
                self.stds[0]= self.stds[0]+ (Xtrain[i][:self.numfeatures]- self.means[0])**2
            if ytrain[i] == 1.0:
                self.stds[1]= self.stds[1]+ (Xtrain[i][:self.numfeatures]- self.means[1])**2

        for i in range(self.numfeatures):
            self.stds[0][i]= (self.stds[0][i]/num_C_0)**0.5
            self.stds[1][i]= (self.stds[1][i]/num_C_1)**0.5
        
        assert self.means.shape == origin_shape
        assert self.stds.shape == origin_shape
    
    def Guassian_prob_dist(self, mean, std, x):
        
        #when the standard deviation is 0 and mean is equal to x, then the probability will always be equal to 1
        #for the inclusion of a column of ones
        if std==0.0 and mean==x:
            return 1
        prob=(1/np.sqrt(2*np.pi * (std**2))) * np.exp((-1/(2*(std**2))) * (x-mean)**2) 
        return prob
    
    def predict(self, Xtest):
        numsamples = Xtest.shape[0]
        predictions = []
        predictions=np.zeros(numsamples, dtype=int)
        
        for i in range(len(predictions)):
            prob_C_0=1
            prob_C_1=1
            
            for j in range(self.numfeatures):
                prob_C_0= prob_C_0 * utils.gaussian_pdf(Xtest[i][j], self.means[0][j], self.stds[0][j])
                prob_C_1= prob_C_1 * utils.gaussian_pdf(Xtest[i][j], self.means[1][j], self.stds[1][j])
            
            prob_C_0=self.p_0 * prob_C_0
            prob_C_1=self.p_1 * prob_C_1
            
            if prob_C_0 > prob_C_1 :
                predictions[i]=0
            elif prob_C_1 >= prob_C_0 :
                predictions[i]=1
            
        assert len(predictions) == Xtest.shape[0]
        return np.reshape(predictions, [numsamples, 1])

#Ans to the Q.no - 1(b)
# Susy: ~23 error
class LogisticReg(Classifier):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({'stepsize': 0.01, 'epochs': 100}, parameters)
        self.weights = None
    
    #this function computes gradient of the cost for logistic regression with respect to theta
    def logit_cost_grad(self,theta,X,y):
        
        grad= np.zeros(1)
        f=utils.sigmoid(np.dot(X.T,theta))
        grad=f-y  
        return grad

    def learn(self, Xtrain, ytrain):
        no_feat = Xtrain.shape[1]
        numsamples=Xtrain.shape[0]
        self.weights=np.random.rand(no_feat)
        epochs=100
        stepsize=self.params['stepsize']
        for i in range(epochs):
            #shuffling data points from 1 to the number of samples n
            datapoints_array=np.arange(numsamples)
            np.random.shuffle(datapoints_array)
            for j in datapoints_array:
                p=np.subtract(utils.sigmoid(np.dot(Xtrain[j].T,self.weights)),ytrain[j])
                gradient=p*Xtrain[j]
                self.weights = self.weights - stepsize* gradient
                
    def predict(self, Xtest):
        numsamples = Xtest.shape[0]
        ytest=np.zeros(numsamples, dtype=int)
        p=utils.sigmoid(np.dot(self.weights,Xtest.T))
        
        for i in range(Xtest.shape[0]):
            if p[i] >=0.5:
                ytest[i]=1.0
            else:
                ytest[i]=0.0
        
        assert len(ytest) == Xtest.shape[0]
        return ytest
        

# Susy: ~23 error (4 hidden units)
class NeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.01,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.wi = None   #W(2) in notes
        self.wo = None   #W(1) in notes

    #feedforward
    def evaluate(self, inputs):
        # hidden activations
        #ah = self.transfer(np.dot(self.wi,inputs.T)) 
        ah = self.transfer(np.dot(self.wi,inputs)) #h in the notes

        # output activations
        #ao = self.transfer(np.dot(self.wo,ah)).T
        ao = self.transfer(np.dot(self.wo,ah))

        return (
            ah, # shape: [nh, samples]
            ao, # shape: [classes, samples]
        )
    
    #backprop algorithm
    def update(self, inputs, outputs):
        h, y_hat = self.evaluate(inputs)
        d_1=y_hat-outputs
        
        d_2 = np.zeros(self.params['nh'])
        d_wo = np.zeros((1,self.params['nh']))
        d_wi = np.zeros((self.params['nh'], self.numfeatures))
        
        for i in range (self.params['nh']):
            d_wo[0][i] = d_1 * h[i]
        
        for i in range (self.params['nh']):
            d_2[i] = (self.wo[0][i] * d_1) * h[i] * (1-h[i])
            d_wi[i] = np.dot(d_2[i], inputs)   
        return (d_wi, d_wo)
    
    def learn(self, Xtrain, ytrain):
        
        self.numfeatures = Xtrain.shape[1]
        self.wi = np.random.randn(self.params['nh'], self.numfeatures)
        self.wo = np.random.randn(1, self.params['nh'])
        
        epochs=self.params['epochs']
        numsamples=Xtrain.shape[0]
        stepsize=self.params['stepsize']
        for i in range(epochs):
            #shuffling data points from 1 to the number of samples n
            datapoints_array=np.arange(numsamples)
            np.random.shuffle(datapoints_array)
            for j in datapoints_array:
                gradient1,gradient2= self.update(Xtrain[j],ytrain[j])
                self.wi = self.wi - stepsize* gradient1
                self.wo = self.wo - stepsize* gradient2
                
    def predict(self,Xtest):
        numsamples = Xtest.shape[0]
        ytest=np.zeros(numsamples, dtype=int)

        for i in range(Xtest.shape[0]):
            h,y=self.evaluate(Xtest[i])
            if y >=0.5:
                ytest[i]=1.0
            else:
                ytest[i]=0.0
        
        assert len(ytest) == Xtest.shape[0]
        return ytest
    
class NeuralNetWithTwoHiddenLayers(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'nh1': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.01,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.w3 = None   #W(3) 
        self.w2 = None   #W(2) 
        self.w1 = None   #W(1) 

    #feedforward
    def evaluate(self, inputs):
        # hidden activations
        ah1 = self.transfer(np.dot(self.w3,inputs)) #h in the notes
        ah2 = self.transfer(np.dot(self.w2,ah1))
        
        # output activations
        ao = self.transfer(np.dot(self.w1,ah2))

        return (
            ah1, # shape: [nh, samples]
            ah2,
            ao, # shape: [classes, samples]
        )
    
    #backprop algorithm
    def update(self, inputs, outputs):
        ah1,ah2,ao = self.evaluate(inputs)
    
        Winput=np.zeros((1,self.params['nh']))
        Wmiddle=np.zeros((self.params['nh'], self.params['nh1']))
        Woutput=np.zeros((self.params['nh1'], self.numfeatures))
        
        d_o=ao-outputs
        Woutput=np.outer(d_o,ah2)
        
        
        d_hid_2=np.dot(self.w1.T,d_o)
        for i in range (self.params['nh']):
            d_hid_2[i] =  d_hid_2[i] * ah2[i] * (1-ah2[i]) 
        
        Wmiddle=np.outer(d_hid_2,ah1)
        
        
        d_hid_1=np.dot(self.w2.T,d_hid_2)
        for i in range (self.params['nh1']):
            d_hid_1[i] =  d_hid_1[i] * ah1[i] * (1-ah1[i]) 
            
        Winput=np.outer(d_hid_1,inputs)
        
        return (Winput, Wmiddle, Woutput)
    
    def learn(self, Xtrain, ytrain):
        
        self.numfeatures = Xtrain.shape[1]
        self.w3= np.random.randn(self.params['nh1'], self.numfeatures)
        self.w2 = np.random.randn(self.params['nh'], self.params['nh1'])
        self.w1 = np.random.randn(1, self.params['nh'])
        
        epochs=self.params['epochs']
        numsamples=Xtrain.shape[0]
        stepsize=self.params['stepsize']
        for i in range(epochs):
            #shuffling data points from 1 to the number of samples n
            datapoints_array=np.arange(numsamples)
            np.random.shuffle(datapoints_array)
            for j in datapoints_array:
                gradient1,gradient2,gradient3= self.update(Xtrain[j],ytrain[j])
                self.w3 = self.w3 - stepsize* gradient1
                self.w2 = self.w2 - stepsize* gradient2
                self.w1 = self.w1 - stepsize* gradient3
                
    def predict(self,Xtest):
        numsamples = Xtest.shape[0]
        ytest=np.zeros(numsamples, dtype=int)

        for i in range(Xtest.shape[0]):
            h1,h2,y=self.evaluate(Xtest[i])
            if y >=0.5:
                ytest[i]=1.0
            else:
                ytest[i]=0.0
        
        assert len(ytest) == Xtest.shape[0]
        return ytest
    
# Note: high variance in errors! Make sure to run multiple times
# Susy: ~28 error (40 centers)
class KernelLogisticRegression(LogisticReg):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 100,
            'centers': 10,
            'kernel': None
        }, parameters)
        self.weights = None


    def linear(self, x, c):
        k=0
        for i in range(x.shape[0]):
            k=k+x[i]*c[i]
        return k
    
    def hamming(self, x, c):
        k=0 
        for i in range(len(x)):
            if x[i]!= c[i]:
                k=k+1
        return k
    
    
    def kernelrepresentation(self,Xtrain):
        
        Ktrain = np.zeros((Xtrain.shape[0], self.numcenters))

        for i in range (Xtrain.shape[0]):
            for j in range (self.numcenters):
                if self.params['kernel'] == 'linear':
                    Ktrain[i][j] = self.linear(Xtrain[i], self.centers[j])
                elif self.params['kernel'] == 'hamming':
                    Ktrain[i][j] == self.hamming(Xtrain[i], self.centers[j])
        return Ktrain

        

    def learn(self, Xtrain, ytrain):
        Ktrain=None
        
        self.numcenters=self.params['centers']
        X = Xtrain
        np.random.shuffle(X)
        self.centers = X[0:self.numcenters]
        
        stepsize=self.params['stepsize']
        
        Ktrain= self.kernelrepresentation(Xtrain)
        
        self.weights=np.zeros(Ktrain.shape[1],)
        numsamples=Xtrain.shape[0]
        epochs=self.params['epochs']
        for i in range(epochs):
            #shuffling data points from 1 to the number of samples n
            datapoints_array=np.arange(numsamples)
            np.random.shuffle(datapoints_array)
            for j in datapoints_array:
                p=np.subtract(utils.sigmoid(np.dot(Ktrain[j],self.weights)),ytrain[j])
                gradient=p*Ktrain[j]
                self.weights = self.weights - stepsize* gradient
    
    def predict(self, Xtest):
        numsamples = Xtest.shape[0]
        ytest=np.zeros(numsamples, dtype=int)
        Ktest=self.kernelrepresentation(Xtest)
        p=utils.sigmoid(np.dot(Ktest,self.weights))
        
        for i in range(Xtest.shape[0]):
            if p[i] >=0.5:
                ytest[i]=1.0
            else:
                ytest[i]=0.0
        
        assert len(ytest) == Xtest.shape[0]
        return ytest
    
class NeuralNetWithTwoHiddenLayersUsingADAM(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'nh1':4,
            'transfer': 'sigmoid',
            'stepsize': 0.01,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.w3 = None   #W(3) 
        self.w2 = None   #W(2) 
        self.w1 = None   #W(1) 
        
        # initialize the values of the parameters
        self.alpha = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.999  
        self.epsilon = 1e-8

    #feedforward
    def evaluate(self, inputs):
        # hidden activations
        ah1 = self.transfer(np.dot(self.w3,inputs)) 
        ah2 = self.transfer(np.dot(self.w2,ah1))
        
        # output activations
        ao = self.transfer(np.dot(self.w1,ah2))

        return (
            ah1, 
            ah2,
            ao, 
        )
    
    #backprop algorithm
    def update(self, inputs, outputs):
        ah1,ah2,ao = self.evaluate(inputs)
        
        Winput=np.zeros((1,self.params['nh']))
        Wmiddle=np.zeros((self.params['nh'], self.params['nh1']))
        Woutput=np.zeros((self.params['nh1'], self.numfeatures))
        
        d_o=ao-outputs
        Woutput=np.outer(d_o,ah2)
       
        d_hid_2=np.dot(self.w1.T,d_o)
        
        for i in range (self.params['nh']):
            d_hid_2[i] =  d_hid_2[i] * ah2[i] * (1-ah2[i]) 
        
        Wmiddle=np.outer(d_hid_2,ah1)
        
        d_hid_1=np.dot(self.w2.T,d_hid_2)
        for i in range (self.params['nh1']):
            d_hid_1[i] =  d_hid_1[i] * ah1[i] * (1-ah1[i]) 
            
        Winput=np.outer(d_hid_1,inputs)
        
        return (Winput, Wmiddle, Woutput)
    
    def learn(self, Xtrain, ytrain):
        
        self.numfeatures = Xtrain.shape[1]
    
        self.w3= np.random.randn(self.params['nh1'], self.numfeatures)
        self.w2 = np.random.randn(self.params['nh'], self.params['nh1'])
        self.w1 = np.random.randn(1, self.params['nh'])
        
        self.m_t_1  = np.zeros((1,self.params['nh']))
        self.m_t_2  = np.zeros((self.params['nh'], self.params['nh']))
        self.m_t_3  = np.zeros((self.params['nh'], self.numfeatures))
        
        self.v_t_1 = np.zeros((1,self.params['nh']))
        self.v_t_2 = np.zeros((self.params['nh'], self.params['nh']))
        self.v_t_3 = np.zeros((self.params['nh'], self.numfeatures))
        
        numsamples=Xtrain.shape[0]
        #stepsize=self.params['stepsize']
        
        t=0
        conv = 0
        while (t<100):					#till 1000 iterations
            t+=1
            for j in range(numsamples):
                gradient1,gradient2,gradient3= self.update(Xtrain[j],ytrain[j])
                
                
                gradient1=(1-self.beta_1)*gradient1
                self.m_t_1=self.beta_1* self.m_t_1
                
                self.m_t_1_temp  = np.zeros((gradient1.shape[0],gradient1.shape[1]))
                
                for i in range(gradient1.shape[0]):
                    for j in range(gradient1.shape[1]):
                        self.m_t_1_temp[i][j]=gradient1[i][j]+self.m_t_1.T[i][0]
                        self.m_t_1_temp[i][j]=self.m_t_1_temp[i][j]/(self.beta_1**t)
                
                
                
                
                gradient2=(1-self.beta_1)*gradient2
                self.m_t_2=self.beta_1* self.m_t_2
                
                self.m_t_2_temp  = np.zeros((gradient2.shape[0],gradient2.shape[1]))
                
                for i in range(gradient2.shape[0]):
                    for j in range(gradient2.shape[1]):
                        self.m_t_2_temp[i][j]=gradient2[i][j]+self.m_t_2.T[i][0]
                        self.m_t_2_temp[i][j]=self.m_t_2_temp[i][j]/(self.beta_1**t)
                
                
                
                gradient3=(1-self.beta_1)*gradient3
                self.m_t_3=self.beta_1* self.m_t_3
                
                self.m_t_3_temp  = np.zeros((gradient3.shape[0],gradient3.shape[1]))
                
                for i in range(gradient3.shape[0]):
                    for j in range(gradient3.shape[1]):
                        self.m_t_3_temp[i][j]=gradient3[i][j]+self.m_t_3.T[i][0]
                        self.m_t_3_temp[i][j]=self.m_t_3_temp[i][j]/(self.beta_1**t)
                
                
                
            
                
                self.v_t_1 = self.beta_2* self.v_t_1 + (1-self.beta_2)*np.dot(gradient1,gradient1.T)
                v_t_1_cap= self.v_t_1/(1-(self.beta_2**t))
                self.v_t_2 = self.beta_2* self.v_t_2 + (1-self.beta_2)*np.dot(gradient2,gradient2.T)
                v_t_2_cap= self.v_t_2/(1-(self.beta_2**t))
                self.v_t_3 = self.beta_2* self.v_t_3 + (1-self.beta_2)*np.dot(gradient3,gradient3.T)
                v_t_3_cap= self.v_t_3/(1-(self.beta_2**t))
                
                weight_prev_1 = self.w1
                weight_prev_2 = self.w2
                weight_prev_3 = self.w3
                self.w1 = weight_prev_1 - (self.alpha*m_t_3_temp)/(np.sqrt(v_t_3_cap)+self.epsilon)	#updates the parameters
                self.w2 = weight_prev_2 - (self.alpha*m_t_2_temp)/(np.sqrt(v_t_2_cap)+self.epsilon)	#updates the parameters
                self.w3 = weight_prev_3 - (self.alpha*m_t_1_temp)/(np.sqrt(v_t_1_cap)+self.epsilon)	#updates the parameters

                
    def predict(self,Xtest):
        numsamples = Xtest.shape[0]
        ytest=np.zeros(numsamples, dtype=int)

        for i in range(Xtest.shape[0]):
            h1,h2,y=self.evaluate(Xtest[i])
            if y >=0.5:
                ytest[i]=1.0
            else:
                ytest[i]=0.0
        
        assert len(ytest) == Xtest.shape[0]
        return ytest

