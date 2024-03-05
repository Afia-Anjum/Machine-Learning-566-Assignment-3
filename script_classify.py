import numpy as np

import MLCourse.dataloader as dtl
import MLCourse.utilities as utils
import classalgorithms as algs
from random import randrange
import math

def getaccuracy(ytest, predictions):
    correct = 0
    # count number of correct predictions
    #correct = np.sum(ytest == predictions)
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    # return percent correct
    return (correct / float(len(ytest))) * 100

def geterror(ytest, predictions):
    #print(ytest.shape)
    #print(predictions.shape)
    return (100 - getaccuracy(ytest, predictions))

""" k-fold cross-validation
K - number of folds
X - data to partition
Y - targets to partition
Algorithm - the algorithm class to instantiate
parameters - a list of parameter dictionaries to test

NOTE: utils.leaveOneOut will likely be useful for this problem.
Check utilities.py for example usage.
"""

def cross_validate(K, X, Y, Algorithm, parameters):
    all_errors = np.zeros((len(parameters), K))
    arr = np.arange(K)
    X_fold = np.split(X, K)
    Y_fold = np.split(Y, K)
    besterror=100
    for k in range(K):

        trainfolds = utils.leaveOneOut(arr, k)

        for i, params in enumerate(parameters):
            
            Larner = Algorithm(params)
            for j in trainfolds:
                Larner.learn(X_fold[j], Y_fold[j])
            predictions = Larner.predict(X_fold[k])
            all_errors[i][k] = geterror(Y_fold[k], predictions)
    
    avg_errors = np.mean(all_errors, axis=1)
    
    
    for i, params in enumerate(parameters):
        #avg_errors[i] = np.mean(all_errors[i]);
        if(avg_errors[i] < besterror):
            besterror=avg_errors[i]
            p=params
            index=i
            
        print('Cross validate parameters:', params)
        print('average error:', avg_errors[i])

    print('Best error over cross validation:', besterror)
    print(p)
    print(str(index))
    best_parameters = parameters[index]
    return best_parameters

def stratified_cross_validate(K, X, Y, Algorithm, parameters):
    
    all_errors = np.zeros((len(parameters), K))
    
    X0=[]
    X1=[]
    Y0=[]
    Y1=[]
    
    num_C_0=0
    num_C_1=0
    
    for i in range(len(Y)):
            if Y[i] == 0.0:
                num_C_0+=1
                Y0.append(Y[i])
                X0.append(X[i])
                
            if Y[i] == 1.0:
                num_C_1+=1
                Y1.append(Y[i])
                X1.append(X[i])
    #spliting each class lebeled dataset into K folds
    X0_fold = np.split(X_0, K)
    Y0_fold = np.split(Y_0, K)
    X1_fold = np.split(X_1, K)
    Y1_fold = np.split(Y_1, K)


    for k in range(K):

        trainfolds = utils.leaveOneOut(arr, k)
        #X_validate and Y_validate hold the validation fold of both class lebel
        X_validate = np.empty(shape=[0, X.shape[1]])
        Y_validate = np.empty([0, ])

        for i, params in enumerate(parameters):
            Learner = Algorithm(params)
            X_train = np.empty(shape=[0, X.shape[1]])
            Y_train = np.empty([0, ])
            #take one fold of class 0 and one fold of class 1
            X_validate = np.concatenate([X_validate, X0_fold[k]])
            Y_validate = np.concatenate([Y_validate, Y0_fold[k]])
            X_validate = np.concatenate([X_validate, X1_fold[k]])
            Y_validate = np.concatenate([Y_validate, Y1_fold[k]])

            # concatenation of k-1 folds
            for j in trainfolds:
                #take remaining folds of both class label as train set
                X_train = np.concatenate([X_train, X0_fold[j]])
                Y_train = np.concatenate([Y_train, Y0_fold[j]])
                X_train = np.concatenate([X_train, X1_fold[j]])
                Y_train = np.concatenate([Y_train, Y1_fold[j]])
                #print(X_train.shape, Y_train.shape)
            #train on train set
            Learner.learn(X_train, Y_train)  # learning on remaining folds other than k
            #predict on validation set
            predictions = Learner.predict(X_validate)
            all_errors[i][k] = geterror(Y_validate, predictions)
            print('error for ' + str(params) + ' on cv fold:' + str(k) + ': ' + str(all_errors[i][k]))

    avg_errors = np.mean(all_errors, axis=1)
    best_parameters = parameters[0]
    best_error = avg_errors[0]
    for i, params in enumerate(parameters):
        avg_errors[i] = np.mean(all_errors[i])
        print('Cross validate parameters:', params)
        print('average error:', avg_errors[i])
        if avg_errors[i] < best_error:
            best_error = avg_errors[i]
            best_parameters = params

    print('Best Parameter', best_parameters)
    return best_parameters




if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Arguments for running.')
    parser.add_argument('--trainsize', type=int, default=5000,
                        help='Specify the train set size')
    parser.add_argument('--testsize', type=int, default=5000,
                        help='Specify the test set size')
    parser.add_argument('--numruns', type=int, default=5,
                        help='Specify the number of runs')
    parser.add_argument('--dataset', type=str, default="susy",
                        help='Specify the name of the dataset')

    args = parser.parse_args()
    trainsize = args.trainsize
    testsize = args.testsize
    numruns = args.numruns
    dataset = args.dataset



    classalgs = {
        'Random': algs.Classifier,
        'Naive Bayes': algs.NaiveBayes,
        'Linear Regression': algs.LinearRegressionClass,
        'Logistic Regression': algs.LogisticReg,
        'Neural Network': algs.NeuralNet,
        'Kernel Logistic Regression': algs.KernelLogisticRegression,
        'NeuralNet_2_Hid_Lay':algs.NeuralNetWithTwoHiddenLayers,
        'NeuralNet_2_Hid_Lay_ADAM':algs.NeuralNetWithTwoHiddenLayersUsingADAM
    }
    numalgs = len(classalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        # name of the algorithm to run
        'Naive Bayes': [
            # first set of parameters to try
            { 'usecolumnones': True },
            # second set of parameters to try
            { 'usecolumnones': False },
        ],
        'Logistic Regression': [
            { 'stepsize': 0.001 },
            { 'stepsize': 0.01 },
            { 'stepsize': 0.1 }
        ],
        'Neural Network': [
            { 'epochs': 100, 'nh': 4 },
            { 'epochs': 100, 'nh': 8 },
            { 'epochs': 100, 'nh': 16 },
            { 'epochs': 100, 'nh': 32 },
        ],
        'Kernel Logistic Regression': [
            { 'centers': 10, 'stepsize': 0.01,'kernel':'linear' },
            #{ 'centers': 10, 'stepsize': 0.01,'kernel':'hamming' },
            { 'centers': 20, 'stepsize': 0.01,'kernel':'linear' },
            #{ 'centers': 20, 'stepsize': 0.01,'kernel':'hamming' },
            { 'centers': 40, 'stepsize': 0.01,'kernel':'linear' },
            #{ 'centers': 40, 'stepsize': 0.01,'kernel':'hamming' },
            { 'centers': 80, 'stepsize': 0.01, 'kernel':'linear' },
            #{ 'centers': 80, 'stepsize': 0.01, 'kernel':'hamming'},
        ],
        'NeuralNet_2_Hid_Lay': [
            #{ 'epochs': 100, 'nh': 4 },
            #{ 'epochs': 100, 'nh': 4,'nh1': 6},
            { 'epochs': 100, 'nh': 8,'nh1': 6 },
            #{ 'epochs': 100, 'nh': 16 },
            #{ 'epochs': 100, 'nh': 32 },
        ],
        'NeuralNet_2_Hid_Lay_ADAM':[
            { 'epochs': 100, 'nh': 4 },
            #{ 'epochs': 100, 'nh': 8 },
            #{ 'epochs': 100, 'nh': 16 },
            #{ 'epochs': 100, 'nh': 32 },
        ]
    }

    # initialize the errors for each parameter setting to 0
    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros(numruns)
    
    for r in range(numruns):
        if dataset == "susy":
            trainset, testset = dtl.load_susy(trainsize, testsize)
        elif dataset == "census":
            trainset, testset = dtl.load_census(trainsize,testsize)
        else:
            raise ValueError("dataset %s unknown" % dataset)

        #print(len(trainset[0]))
        Xtrain = trainset[0]
        Ytrain = trainset[1]
        #print(Xtrain.shape)
        
        # cast the Y vector as a matrix
        Ytrain = np.reshape(Ytrain, [len(Ytrain), 1])
        #print(Ytrain.shape)

        Xtest = testset[0]
        Ytest = testset[1]
        # cast the Y vector as a matrix
        Ytest = np.reshape(Ytest, [len(Ytest), 1])
        best_parameters = {}
        for learnername, Learner in classalgs.items():
            params = parameters.get(learnername, [ None ])
            #best_parameters[learnername] = stratified_cross_validate(5, Xtrain, Ytrain, Learner, params)
            best_parameters[learnername] = cross_validate(5, Xtrain, Ytrain, Learner, params)

        for learnername, Learner in classalgs.items():
            params = best_parameters[learnername]
            #params = parameters.get(learnername, [ None ])
            learner = Learner(params)
            
            #Ytrain = trainset[1]
            
            learner.learn(Xtrain,Ytrain)
            prediction=learner.predict(Xtest)
            
            #Ytest = testset[1]
            
            #Ytest = np.reshape(Ytest, [len(Ytest), 1])
            error=geterror(Ytest, prediction)
            errors[learnername][r]=error
        
        #print(errors)
    for learnername in classalgs:
        aveerror = np.mean(errors[learnername])
        print('Average error for ' + learnername + ': ' + str(aveerror))
        stand_error=np.std(errors[learnername])/math.sqrt(numruns)
        print('Standard error for ' + learnername + ': ' + str(stand_error))
        best=best_parameters[learnername]
        print ('Meta parameters chosen by Cross Validation for ' + learnername + ': ' + str(best))




