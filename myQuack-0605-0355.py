
'''

Some partially defined functions for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls different functions to perform the required tasks.

'''
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Import the required modules


import numpy as np

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import tree
from sklearn import svm
import matplotlib.pyplot as plt

import time


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Global variables and constants
NUM_OF_COL = 32
LABEL_COL = 2

VALIDATION_PERCENTAGE = 0.3

MIN_SPLITS = 2
MAX_SPLITS = int(-235*VALIDATION_PERCENTAGE + 222.5) # a function of VALIDATION_PERCENTAGE

K = [2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15]
# lists for storing data to make tables and figures with
nb_classifier_results = []
nb_classifier_results.append(['k','score'])

dt_classifier_results = []
dt_classifier_results.append(['k', 'max_depth', 'clf_score'])

nn_classifier_results = []
nn_classifier_results.append(['k','n_neighbors','score'])

svm_classifier_results = []
nb_classifier_results.append(['k', 'kernel', 'C', 'gamma', 'clf_score'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def mean_cross_val(clf, X, y, k):
    '''
    Cross validates the classifier and returns the mean score
        Uses np.float64 for greater accuracy
    '''
    return np.mean(cross_val_score(clf, X, y, cv = k), dtype = np.float64)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (9370331, 'Jiaming', 'Chen'), (9713581, 'Christopher Ayling') ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    ##         "INSERT YOUR CODE HERE"    
    data_col = list(range(NUM_OF_COL))
    data_col.remove(LABEL_COL-1)
    data_col.remove(0)

    data = np.genfromtxt(dataset_path,dtype=None,delimiter=',', usecols=set(data_col))    
    data_label = np.genfromtxt(dataset_path,dtype=None,delimiter=',', usecols=LABEL_COL-1)   
    data_label_list = []
    for i in list(range(len(data_label))):
        if data_label[i] == b'M':
            data_label_list.append(1)
        else:
            data_label_list.append(0)
            
    return data, np.array(data_label_list)    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"

    NB_list = []
    #initialise best clf
    best_clf = GaussianNB()
    best_clf.fit(X_training, y_training)
    best_clf_score = mean_cross_val(best_clf, X_training, y_training, k = MIN_SPLITS)    
        
    #loop for changing cross validation value
    for k in K:
        kfold = KFold(n_splits = k)
        for train_index, test_index in kfold.split(X_training):
            #seperate training and test
            X_train, X_test = X_training[train_index], X_training[test_index]
            y_train, y_test = y_training[train_index], y_training[test_index]        
            #create classifier
            clf = GaussianNB()
            clf.fit(X_train, y_train)
            #cross validate
            clf_score = mean_cross_val(clf, X_test, y_test, k = k)
            #write data
            nb_classifier_results.append([k, clf_score])            
            #if the newly generate clf is better than the previous best,
            #   set it as the new best
            if (clf_score > best_clf_score):
                best_clf = clf
                best_clf_score = clf_score
                print('new best found! score = ', best_clf_score, ' | k = ', k)
            
        NB_list.append([k,clf, clf_score])
    #return the most accurate classifier
    return best_clf, np.array(NB_list)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    DT_list = []
    #initialise best clf
    best_clf = tree.DecisionTreeClassifier()
    best_clf.fit(X_training, y_training)
    best_clf_score = best_clf.score(X_testing, y_testing)
    
    #declare possible parameters
    max_depths = [None]
    for n in range(1, NUM_OF_COL - LABEL_COL + 1):
        max_depths.append(n)
        
    #tune hyper-parameters
    tuned_parameters = [{'max_depth': max_depths}]
    
    #create classifier
    clf = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5)    
        
    #loop for changing cross validation value
    for k in K:
        kfold = KFold(n_splits = k)
        for train_index, test_index in kfold.split(X_training):
            #seperate training and test
            X_train, X_test = X_training[train_index], X_training[test_index]
            y_train, y_test = y_training[train_index], y_training[test_index]
            #create classifier
            clf = tree.DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            #cross validate
            clf_score = mean_cross_val(clf, X_test, y_test, k = k)
            #if the newly generate clf is better than the previous best,
            #   set it as the new best
            if (clf_score > best_clf_score):
                best_clf = clf
                best_clf_score = clf_score
                print('new best found! score = ', clf_score, ' | k = ', k)

        DT_list.append([k,clf, clf_score])
        
    #return the most accurate classifier
    return best_clf, np.array(DT_list)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    
    K = [2, 3, 4, 5]
    ##         "INSERT YOUR CODE HERE"
    NN_list = []
    #initialise best clf
    best_clf = neighbors.KNeighborsClassifier()
    best_clf.fit(X_training, y_training)
    best_clf_score = mean_cross_val(best_clf, X_training, y_training, k = MIN_SPLITS)    
    
    #tune hyper-parameters
    tuned_parameters = [{'n_neighbors': [5, 10, 20, 30, 40]}]
    
    #create classifier
    clf = GridSearchCV(neighbors.KNeighborsClassifier(), tuned_parameters, cv=5)
    
    # try all possible combinations and determine the best
    for k in K:
        kfold = KFold(n_splits = k)
        for train_index, test_index in kfold.split(X_training):
            #seperate training and test
            X_train, X_test = X_training[train_index], X_training[test_index]
            y_train, y_test = y_training[train_index], y_training[test_index]
            #fit classifier
            clf.fit(X_train, y_train)
            #cross validate
            clf_score = mean_cross_val(clf, X_test, y_test, k = k)
            #if the newly generate clf is better than the previous best,
            #   set it as the new best
            if (clf_score > best_clf_score):
                best_clf = clf
                best_clf_score = clf_score
                print('new best found! score = ', best_clf_score, ' | k = ', k)
        NN_list.append([k,clf, clf_score])
        
    #return the most accurate classifier
    return best_clf, np.array(NN_list)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    SVM_list = []
    #initialise best clf
    best_clf = svm.SVC()
    best_clf.fit(X_training, y_training)
    best_clf_score = best_clf.score(X_testing, y_testing)
    
    #declare possible parameters
    gammas = [1000, 100, 10, 0.1, 0.001, 0.0001]
    
    #tune hyper-parameters
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gammas, 'C': [1]}]
    
    #create classifier
    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5)

    #try all possible combinations and return the best one
    for k in K:
        kfold = KFold(n_splits = k)
        for train_index, test_index in kfold.split(X_training):
            #seperate training and test
            X_train, X_test = X_training[train_index], X_training[test_index]
            y_train, y_test = y_training[train_index], y_training[test_index]            
            #fit classifier
            clf.fit(X_train, y_train)
            #cross validate
            clf_score = mean_cross_val(clf, X_test, y_test, k = k)
            #if the newly generate clf is better than the previous best,
            #   set it as the new best
            if (clf_score > best_clf_score):
                best_clf = clf
                best_clf_score = clf_score
                print('new best found! score = ', best_clf_score, ' | k = ', k)
        SVM_list.append([k,clf, clf_score])
        
    #return the most accurate classifier
    return best_clf, np.array(SVM_list)
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def test_model_and_show_figure(np_classifier, classifier_name):
    testing_list = []
    for i in list(range(np_classifier.shape[0])):
        testing_list.append(np_classifier[i][1].score(X_testing, y_testing))
    np_testing = np.array(testing_list)
    
    plt.plot(np_classifier[:,0], 1-np_classifier[:,2],'b-o',label="validation")
    plt.plot(np_classifier[:,0], 1-np_testing[:],'r-o',label="test")
    plt.legend(loc='upper left')
    plt.ylabel('Error Rate(%)')
    plt.xlabel('K')
    plt.title(classifier_name)
    plt.show()  


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if __name__ == "__main__":

    print('\n# - - - PREPROCESSING - - - #')
    
    # preprocessing    
    dataset_path = "medical_records.data"
    X_dataset, y_dataset = prepare_dataset(dataset_path)
    
    # determine ratio of training to validation
    len_testingset = int(len(X_dataset)*VALIDATION_PERCENTAGE)
    print('\ntraining:validation = ', 1 - VALIDATION_PERCENTAGE,':',VALIDATION_PERCENTAGE)
    print('\nMIN_SPLITS = ', MIN_SPLITS)
    print('MAX_SPLITS = ', MAX_SPLITS)

    
    # X/y_training is for training
    # X/y_testing is for validation
    X_training, X_testing = X_dataset[:len(X_dataset)-len_testingset], X_dataset[len(X_dataset)-len_testingset:]
    y_training, y_testing = y_dataset[:len(X_dataset)-len_testingset], y_dataset[len(X_dataset)-len_testingset:]



    print('\n# - - - BUILDING CLASSIFIERS - - - #')
    
    #list for storing execution times
    exec_times = []
    
    # build NB classifier
    print('\nBuilding NB Classifier...')
    t1 = time.clock()
    NB,np_NB = build_NB_classifier(X_training, y_training)
    t2 = time.clock()
    exec_times.append(t2-t1)
    test_model_and_show_figure(np_NB, "Naive Bayer")
    
    # build DT classifier
    print('\nBuilding DT Classifier...')
    t1 = time.clock()
    DT,np_DT = build_DT_classifier(X_training, y_training)
    t2 = time.clock()
    exec_times.append(t2-t1)
    test_model_and_show_figure(np_DT, "Decision Tree")
    
    # build KNN classifier
    print('\nBuilding KNN Classifier...')
    t1 = time.clock()
    KNN,np_KNN = build_NN_classifier(X_training, y_training)
    t2 = time.clock()
    exec_times.append(t2-t1)
    test_model_and_show_figure(np_KNN, "KNN")
    
    # build SVM classifier
    print('\nBuilding SVM Classifier...')
    t1 = time.clock()
    SVM,np_SVM = build_SVM_classifier(X_training, y_training)
    t2 = time.clock()
    exec_times.append(t2-t1)
    test_model_and_show_figure(np_SVM, "SVM")
    
    
    
    print('\n# - - - BUILDING FIGURES AND TABLES - - - #')
    
    # execution times [nb, dt, knn, svm]
    print('\nexecution times: ', exec_times)
    
    # tables and figures

    
    
    print('\n# - - - CLASSIFIER VALIDATION - - - #')
    
    # validation of NB classifier
    print('\nNaive-Bayes Accuracy: ', NB.score(X_testing, y_testing))
    
    # validation of DT classifier
    print('\nDecision Tree Accuracy: ', DT.score(X_testing, y_testing))
    
    # validaiton of NN classifier
    print('\nNearest Neighbor Accuracy: ', KNN.score(X_testing, y_testing))
    
    # validation of SVM classifier
    print('\nSupport Vector Machine Accuracy: ', SVM.score(X_testing, y_testing))
