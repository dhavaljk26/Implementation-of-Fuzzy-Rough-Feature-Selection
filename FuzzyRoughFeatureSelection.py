
# coding: utf-8

# In[25]:


import numpy as np
import skfuzzy as fuzz
import math


# In[26]:


# def fuzzify(arr):
#     mem = np.empty((2, arr.shape[0]))
#     mem[0] = fuzz.membership.trapmf(arr, [-1, -1, -0.5, 0])
#     mem[1] = fuzz.membership.trimf(arr, [-0.5, 0, 0.5])
#     return np.transpose(mem)

# def fuzzify_data(data):
#     data_transpose = np.transpose(data)
#     fuzzy_data_transpose = np.empty((data_transpose.shape[0], data_transpose.shape[1], 2))

#     for i in range(0, data_transpose.shape[0]):
#         fuzzy_data_transpose[i] = fuzzify(data_transpose[i])
        
#     fuzzy_data = np.transpose(fuzzy_data_transpose)  
#     return fuzzy_data   


# In[27]:


def t_norm(x, y):
    return max(x+y-1.0, 0.0)

def fuzzy_implicator(x, y):
    return min(1.0-x+y, 1.0)


# In[28]:


def attribute_fuzzy_similatity_measure(arr):
    
    arr_2 = np.zeros( (arr.shape[0], arr.shape[0]) )
    sigma = math.sqrt((np.var(arr, ddof=1)))
    
    if sigma == 0.0:
        return np.ones( (arr.shape[0], arr.shape[0]) )
    
    for x in range(arr.shape[0]):
        for y in range(arr.shape[0]):
            arr_2[x][y] = arr_2[y][x] = max(0.0, min( float(arr[y]-arr[x]+sigma)/sigma, float(arr[x]+sigma-arr[y])/sigma))

    for i in range(arr_2.shape[0]):
        for j in range(arr_2.shape[0]):
            for k in range(arr_2.shape[0]):
                arr_2[j][k] = max(arr_2[j][k], t_norm(arr_2[j][i], arr_2[i][k]))
    return arr_2


# In[29]:


def fuzzy_similatity_measure(data):
    fsm = np.zeros( (data.shape[1]-1, data.shape[0], data.shape[0]) )
    data_transpose = np.transpose(data)
    
    for i in range(data.shape[1]-1):
        fsm[i] = attribute_fuzzy_similatity_measure(data_transpose[i])
    
    return fsm


# In[30]:


data = np.array([
        [-0.4, -0.3, -0.5, 0], 
        [-0.4,  0.2, -0.1, 1], 
        [-0.3, -0.4, -0.3, 0], 
        [ 0.3, -0.3,  0.0, 1], 
        [ 0.2, -0.3,  0.0, 1], 
        [ 0.2,  0.0,  0.0, 0]
        ])


# In[31]:


def equivalence_class(data, decision_features):
    E = dict()
    s = data.shape
    
    for i in np.unique(np.transpose(data)[-1]):
        E[int(i)] = set()
        
    for i in range(0, s[0]):
        E[ int(data[i][s[1]-1]) ].add(i)
                       
    return E


# In[32]:


def present_in_set(s, element):
    if element in s:
        return 1
    else: 
        return 0


# In[33]:


def dependency_measurement(data, conditional_features_set, equivalence_class, fsm, mode):
    
    mu_r = np.ones( (data.shape[0], data.shape[0]) )
    
    for i in conditional_features_set:
        for x in range(data.shape[0]):
            for y in range(data.shape[0]):
                mu_r[x][y] = t_norm(mu_r[x][y], fsm[i][x][y])
                
    mu = np.zeros( (data.shape[0]) )
    
    for x in range(data.shape[0]):
        for class_, eq_class in equivalence_class.items():
            I = 1.0
            for y in range(data.shape[0]):
                I = min(I, fuzzy_implicator(mu_r[x][y], present_in_set(eq_class, y))) 
            mu[x] = max(I, mu[x])    
    return np.sum(mu)/mu.size


# In[34]:


def fuzzy_rough_quick_reduct(data, conditional_features, decision_features, mode='lower'):
    
    C = set(conditional_features)
    D = set(decision_features)
    R = set()
    lambda_best = 0.0
    lambda_prev = 0.00000001

    fsm = fuzzy_similatity_measure(data)
    equivalence_class_map = equivalence_class(data, decision_features)

    while lambda_best != lambda_prev:
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
         
        if lambda_max > lambda_T:
            print (x, lambda_max, lambda_best)
            T.add(x)
            lambda_best = lambda_max
            
        R = T.copy()
    return R    
        


# In[35]:


r = fuzzy_rough_quick_reduct(data, np.arange(data.shape[1]-1), np.array([data.shape[1]-1]), fsm)
print(r)

