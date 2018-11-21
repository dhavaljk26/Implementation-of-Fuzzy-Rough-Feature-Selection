
# coding: utf-8

# In[209]:


#Import Libraries 

import numpy as np
import skfuzzy as fuzz
import math
import pandas as pd
from sklearn.model_selection import train_test_split


# In[210]:


#Calculate  Łukasiewicz t_norm and  Łukasiewicz fuzzy_implicator, as given in 
# [Author: Richard Jensen, Qiang Shen, 
#  Paper : New Approaches to Fuzzy-Rough Feature Selection, 
#  Link  : https://ieeexplore.ieee.org/document/4505335]

def t_norm(x, y):
    return max(x+y-1.0, 0.0)

def fuzzy_implicator(x, y):
    return min(1.0-x+y, 1.0)


# In[211]:


#Calculate fuzzy similarity measure for a particular attribute, as given in 
# [Author: Richard Jensen, Qiang Shen, 
#  Paper : New Approaches to Fuzzy-Rough Feature Selection, 
#  Link  : https://ieeexplore.ieee.org/document/4505335]

def attribute_fuzzy_similarity_measure(arr):
    
    arr_2 = np.zeros( (arr.shape[0], arr.shape[0]) )
    sigma = math.sqrt((np.var(arr, ddof=1)))
    
    if sigma == 0.0:
        return np.ones( (arr.shape[0], arr.shape[0]) )
    
    for x in range(arr.shape[0]):
        for y in range(arr.shape[0]):
            arr_2[x][y] = arr_2[y][x] = max(0.0, min( float(arr[y]-arr[x]+sigma)/sigma, float(arr[x]+sigma-arr[y])/sigma))

#     for i in range(arr_2.shape[0]):
#         for j in range(arr_2.shape[0]):
#             for k in range(arr_2.shape[0]):
#                 arr_2[j][k] = max(arr_2[j][k], t_norm(arr_2[j][i], arr_2[i][k]))
    return arr_2


# In[212]:


#Calculate fuzzy similarity measure for a dataset for all attributes

def fuzzy_similatity_measure(data):
    fsm = np.zeros( (data.shape[1]-1, data.shape[0], data.shape[0]) )
    data_transpose = np.transpose(data)
    
    for i in range(data.shape[1]-1):
        fsm[i] = attribute_fuzzy_similarity_measure(data_transpose[i])
    
    return fsm


# In[213]:


#Calculate equivalence classes based on decision features

def equivalence_class(data, decision_features):
    E = dict()
    s = data.shape
    
    for i in np.unique(np.transpose(data)[-1]):
        E[int(i)] = set()
        
    for i in range(0, s[0]):
        E[ int(data[i][s[1]-1]) ].add(i)
                       
    return E


# In[214]:


#Check if element is present in a set

def present_in_set(s, element):
    if element in s:
        return 1
    else: 
        return 0


# In[215]:


# Calculate dependency measure of conditional features on data based on equivalence classes
# [Note: dependency_measure is different for different modes, as given in paper]

def dependency_measurement(data, conditional_features_set, equivalence_class, fsm, mode):
    
    mu_r = np.ones( (data.shape[0], data.shape[0]) )
    
    for i in conditional_features_set:
        for x in range(data.shape[0]):
            for y in range(data.shape[0]):
                mu_r[x][y] = t_norm(mu_r[x][y], fsm[i][x][y])
                
    mu = np.zeros( (data.shape[0]) )
    
    if mode == 'boundary':
        uncertainty = 0.0
        for class_, eq_class in equivalence_class.items():
            
            for x in range(data.shape[0]):
                I = 1.0
                T = 0.0
                for y in range(data.shape[0]):
                    I = min(I, fuzzy_implicator(mu_r[x][y], present_in_set(eq_class, y))) 
                    T = max(T, t_norm(mu_r[x][y], present_in_set(eq_class, y)))
                mu[x] = T - I
            uncertainty += (np.sum(mu)/mu.size)/len(equivalence_class)
        return 1.0 - uncertainty
    else:
        
        for x in range(data.shape[0]):
            for class_, eq_class in equivalence_class.items():
                I = 1.0
                for y in range(data.shape[0]):
                    I = min(I, fuzzy_implicator(mu_r[x][y], present_in_set(eq_class, y))) 

                mu[x] = max(I, mu[x])  

        return np.sum(mu)/mu.size


# In[216]:


# Fuzzy-Rough Quick Reduct as given in paper
# [Note: Two modes are
#        lower    :  Fuzzy Lower Approximation-Based FS
#        boundary :  Fuzzy Boundary Region-Based FS]

def fuzzy_rough_quick_reduct(data, conditional_features, decision_features, mode='lower'):
    
    if mode != 'lower' and mode != 'boundary':
        print ("Mode should be from ['lower', 'boundary']")
        return 
    
    C = set(conditional_features)
    D = set(decision_features)
    R = set()
    lambda_best = 0.0
    lambda_prev = 0.00000001

    fsm = fuzzy_similatity_measure(data)
    equivalence_class_map = equivalence_class(data, decision_features)
    while lambda_best != lambda_prev and lambda_best < 1.0:
        T = R.copy()
        lambda_prev = lambda_best
        
        lambda_T = dependency_measurement(data, T, equivalence_class_map, fsm, mode)
        
        lambda_max = lambda_T
        feature_to_add = None
        
        C_minus_R = C.difference(R)
        for x in C_minus_R:
            temp = set()
            temp = R.copy()
            temp.add(x)
            lambda_temp = dependency_measurement(data, temp, equivalence_class_map, fsm, mode)
                        
            if lambda_temp > lambda_max:
                lambda_max = lambda_temp
                feature_to_add = x
         
        if lambda_max > lambda_T and feature_to_add != None:
            T.add(feature_to_add)
            lambda_best = lambda_max
            
        R = T.copy()
    return R    
        


# In[217]:


#Example dataset given in paper (for testing)
example_data = np.array([
        [-0.4, -0.3, -0.5, 0], 
        [-0.4,  0.2, -0.1, 1], 
        [-0.3, -0.4, -0.3, 0], 
        [ 0.3, -0.3,  0.0, 1], 
        [ 0.2, -0.3,  0.0, 1], 
        [ 0.2,  0.0,  0.0, 0]
        ])


# In[218]:


# Preprocessing

#Read from CSV
data = pd.read_csv('new.csv',  header=None)

#Remove missing values
data = data.replace(-1, np.nan)
data = data.dropna()

#Convert dataframe to numpy matrix
data_matrix = data.values


# In[219]:


# Calculate reduct set of attributes

reduct = fuzzy_rough_quick_reduct(data_matrix, np.arange(data_matrix.shape[1]-1), np.array([data_matrix.shape[1]-1]), 'boundary')
print(reduct)


# In[406]:


# Measure accuracy by comparing values of predicted and original values

def measure_accuracy(pred, orig):
    accuracy = 0
    for i in range(len(pred)):
            if pred[i] == orig[i]:
                accuracy+=1
    accuracy /= pred.shape[0]
    
    return accuracy


# In[407]:


# Classification using classifier parameter on whole dataset and dataset with reduct set of attributes

def measure_classification_accuracy(classifier, data, reduct):
    
    c_train, c_test, d_train, d_test = train_test_split(data[:, :-1], data[:, -1:])
    
    C = set(np.arange(data.shape[1]-1))
    rc_train = np.delete(c_train, list(C.difference(reduct)), 1)
    rc_test = np.delete(c_test, list(C.difference(reduct)), 1)
    
    if classifier == 'DecisionTree':
        from sklearn.tree import DecisionTreeClassifier 

        dtree_model = DecisionTreeClassifier(max_depth = 20).fit(c_train, d_train) 
        dtree_predictions = dtree_model.predict(c_test)

        rdtree_model = DecisionTreeClassifier(max_depth = 100).fit(rc_train, d_train) 
        rdtree_predictions = rdtree_model.predict(rc_test)

        return measure_accuracy(dtree_predictions, d_test), measure_accuracy(rdtree_predictions, d_test)
    
    elif classifier == 'SVM':
        from sklearn.svm import SVC 
        svm_model_linear = SVC(kernel = 'linear', C = 1).fit(c_train, d_train.ravel()) 
        svm_predictions = svm_model_linear.predict(c_test) 

        accuracy = svm_model_linear.score(c_test, d_test)
        
        rsvm_model_linear = SVC(kernel = 'linear', C = 1).fit(rc_train, d_train.ravel()) 
        rsvm_predictions = rsvm_model_linear.predict(rc_test) 

        reduct_accuracy = rsvm_model_linear.score(rc_test, d_test)
        
        return accuracy, reduct_accuracy
    
    elif classifier == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier 

        knn = KNeighborsClassifier(n_neighbors = 20).fit(c_train, d_train.ravel()) 
        accuracy = knn.score(c_test, d_test)
        
        rknn = KNeighborsClassifier(n_neighbors = 20).fit(rc_train, d_train.ravel()) 
        reduct_accuracy = rknn.score(rc_test, d_test)
        
        return accuracy, reduct_accuracy
    
    elif classifier == 'GaussianNaiveBayes':
        from sklearn.naive_bayes import GaussianNB 
        
        gnb = GaussianNB().fit(c_train, d_train.ravel()) 
        gnb_predictions = gnb.predict(c_test) 
        
        rgnb = GaussianNB().fit(rc_train, d_train.ravel()) 
        rgnb_predictions = rgnb.predict(rc_test) 
      
        return measure_accuracy(gnb_predictions, d_test), measure_accuracy(rgnb_predictions, d_test)

    elif classifier == 'MultiLayerPerceptron':
        from sklearn.neural_network import MLPClassifier
        
        mlp = MLPClassifier().fit(c_train, d_train.ravel())
        mlp_predictions = mlp.predict(c_test)
        
        rmlp = MLPClassifier().fit(rc_train, d_train.ravel())
        rmlp_predictions = rmlp.predict(rc_test)
        
        return measure_accuracy(mlp_predictions, d_test), measure_accuracy(rmlp_predictions, d_test)
    
    elif classifier == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=10).fit(c_train, d_train.ravel())
        rf_predictions = rf.predict(c_test)
        
        rrf = RandomForestClassifier(n_estimators=10).fit(rc_train, d_train.ravel())
        rrf_predictions = rrf.predict(rc_test)
        
        return measure_accuracy(rf_predictions, d_test), measure_accuracy(rrf_predictions, d_test)
        
    else:
        print ('Invalid parameter(classifier)')
        return None, None
    


# In[411]:



all_accuracy, reduct_accuracy = measure_classification_accuracy('RandomForest', data_matrix, reduct)
print ("All attributes accuracy    = ",all_accuracy, "\nReduct attributes accuracy = ",reduct_accuracy)

